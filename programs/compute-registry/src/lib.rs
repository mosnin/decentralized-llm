use anchor_lang::prelude::*;
use anchor_spl::associated_token::AssociatedToken;
use anchor_spl::token::{self, Mint, Token, TokenAccount, Transfer};

declare_id!("8KpR2mT6uLqVwNzS4eBfY9oA3cJ7iGxH1nD5sW0qF2M");

/// Minimum stake required to register as a compute node (100k tokens, 6 decimals)
const MIN_STAKE: u64 = 100_000 * 1_000_000;
/// Fraction of stake slashed on confirmed misbehavior (10%)
const SLASH_BPS: u64 = 1_000;

#[program]
pub mod compute_registry {
    use super::*;

    pub fn initialize(ctx: Context<Initialize>) -> Result<()> {
        let registry = &mut ctx.accounts.registry;
        registry.authority = ctx.accounts.authority.key();
        registry.token_mint = ctx.accounts.token_mint.key();
        registry.total_nodes = 0;
        registry.bump = ctx.bumps.registry;
        Ok(())
    }

    /// Register a new compute node and lock stake.
    ///
    /// `endpoint`    – libp2p multiaddr or HTTPS URL for P2P connectivity
    /// `vram_gb`     – VRAM capacity (determines which model shards can be hosted)
    /// `gpu_count`   – number of GPUs on this node
    /// `model_ids`   – list of model shard IDs this node will serve (max 8)
    pub fn register_node(
        ctx: Context<RegisterNode>,
        endpoint: String,
        vram_gb: u16,
        gpu_count: u8,
        model_ids: Vec<[u8; 32]>,
        stake_amount: u64,
    ) -> Result<()> {
        require!(endpoint.len() <= 256, RegistryError::EndpointTooLong);
        require!(model_ids.len() <= 8, RegistryError::TooManyModels);
        require!(stake_amount >= MIN_STAKE, RegistryError::InsufficientStake);
        require!(vram_gb > 0, RegistryError::InvalidVram);

        // Lock stake in the node's escrow PDA
        token::transfer(
            CpiContext::new(
                ctx.accounts.token_program.to_account_info(),
                Transfer {
                    from: ctx.accounts.operator_token_account.to_account_info(),
                    to: ctx.accounts.stake_token_account.to_account_info(),
                    authority: ctx.accounts.operator.to_account_info(),
                },
            ),
            stake_amount,
        )?;

        let registry = &mut ctx.accounts.registry;
        let node = &mut ctx.accounts.node;

        node.operator = ctx.accounts.operator.key();
        node.endpoint = endpoint;
        node.vram_gb = vram_gb;
        node.gpu_count = gpu_count;
        node.model_ids = model_ids;
        node.staked_amount = stake_amount;
        node.reputation = 1000; // start at 1000 out of 1000
        node.jobs_completed = 0;
        node.jobs_disputed = 0;
        node.is_active = true;
        node.registered_at = Clock::get()?.unix_timestamp;
        node.bump = ctx.bumps.node;

        registry.total_nodes = registry.total_nodes.checked_add(1).unwrap();

        emit!(NodeRegistered {
            operator: node.operator,
            endpoint: node.endpoint.clone(),
            vram_gb: node.vram_gb,
            stake_amount,
        });

        Ok(())
    }

    /// Node operator adds more stake (increases security bond).
    pub fn add_stake(ctx: Context<ModifyStake>, amount: u64) -> Result<()> {
        require!(amount > 0, RegistryError::ZeroAmount);

        token::transfer(
            CpiContext::new(
                ctx.accounts.token_program.to_account_info(),
                Transfer {
                    from: ctx.accounts.operator_token_account.to_account_info(),
                    to: ctx.accounts.stake_token_account.to_account_info(),
                    authority: ctx.accounts.operator.to_account_info(),
                },
            ),
            amount,
        )?;

        let node = &mut ctx.accounts.node;
        node.staked_amount = node.staked_amount.checked_add(amount).unwrap();

        Ok(())
    }

    /// Node operator initiates unstake. There is a 7-day cooldown before withdrawal
    /// to allow in-flight jobs to complete and disputes to resolve.
    pub fn begin_unstake(ctx: Context<BeginUnstake>, amount: u64) -> Result<()> {
        let node = &mut ctx.accounts.node;

        require!(node.operator == ctx.accounts.operator.key(), RegistryError::NotOperator);
        require!(amount > 0, RegistryError::ZeroAmount);
        require!(
            node.staked_amount.checked_sub(amount).unwrap_or(0) >= MIN_STAKE
                || node.staked_amount == amount,
            RegistryError::WouldDropBelowMin
        );

        node.unstake_amount = amount;
        node.unstake_at = Clock::get()?.unix_timestamp + 7 * 24 * 3600;
        // Mark inactive if fully unstaking
        if node.staked_amount == amount {
            node.is_active = false;
        }

        Ok(())
    }

    /// Withdraw unstaked tokens after cooldown.
    pub fn withdraw_stake(ctx: Context<WithdrawStake>) -> Result<()> {
        let node = &ctx.accounts.node;

        require!(node.operator == ctx.accounts.operator.key(), RegistryError::NotOperator);
        require!(node.unstake_amount > 0, RegistryError::NoPendingUnstake);
        require!(
            Clock::get()?.unix_timestamp >= node.unstake_at,
            RegistryError::CooldownNotMet
        );

        let amount = node.unstake_amount;
        let seeds = &[b"node".as_ref(), node.operator.as_ref(), &[node.bump]];
        let signer = &[&seeds[..]];

        token::transfer(
            CpiContext::new_with_signer(
                ctx.accounts.token_program.to_account_info(),
                Transfer {
                    from: ctx.accounts.stake_token_account.to_account_info(),
                    to: ctx.accounts.operator_token_account.to_account_info(),
                    authority: ctx.accounts.node.to_account_info(),
                },
                signer,
            ),
            amount,
        )?;

        let node = &mut ctx.accounts.node;
        node.staked_amount = node.staked_amount.checked_sub(amount).unwrap();
        node.unstake_amount = 0;
        node.unstake_at = 0;

        Ok(())
    }

    /// Record job completion. Called by the inference-market program via CPI.
    /// Updates reputation and completed job count.
    pub fn record_completion(ctx: Context<RecordCompletion>) -> Result<()> {
        let node = &mut ctx.accounts.node;

        node.jobs_completed = node.jobs_completed.checked_add(1).unwrap();
        // Gradually restore reputation toward 1000 on successful completion
        node.reputation = (node.reputation + 1).min(1000);

        Ok(())
    }

    /// Slash a node's stake after a governance-confirmed dispute resolution.
    /// Only callable by the governance program PDA.
    pub fn slash(ctx: Context<Slash>) -> Result<()> {
        let node = &mut ctx.accounts.node;

        let slash_amount = node
            .staked_amount
            .checked_mul(SLASH_BPS)
            .unwrap()
            .checked_div(10_000)
            .unwrap();

        let seeds = &[b"node".as_ref(), node.operator.as_ref(), &[node.bump]];
        let signer = &[&seeds[..]];

        // Slash goes to DAO treasury
        token::transfer(
            CpiContext::new_with_signer(
                ctx.accounts.token_program.to_account_info(),
                Transfer {
                    from: ctx.accounts.stake_token_account.to_account_info(),
                    to: ctx.accounts.treasury_token_account.to_account_info(),
                    authority: ctx.accounts.node.to_account_info(),
                },
                signer,
            ),
            slash_amount,
        )?;

        node.staked_amount = node.staked_amount.checked_sub(slash_amount).unwrap();
        node.jobs_disputed = node.jobs_disputed.checked_add(1).unwrap();
        // Significant reputation penalty
        node.reputation = node.reputation.saturating_sub(100);

        // Auto-deactivate if reputation drops too low
        if node.reputation < 200 {
            node.is_active = false;
        }

        emit!(NodeSlashed {
            operator: node.operator,
            slash_amount,
            reputation: node.reputation,
        });

        Ok(())
    }

    /// Update node endpoint (e.g., IP changed after GPU re-rental).
    pub fn update_endpoint(ctx: Context<UpdateNode>, endpoint: String) -> Result<()> {
        require!(endpoint.len() <= 256, RegistryError::EndpointTooLong);
        let node = &mut ctx.accounts.node;
        require!(node.operator == ctx.accounts.operator.key(), RegistryError::NotOperator);
        node.endpoint = endpoint;
        Ok(())
    }
}

// ─────────────────────────── Account structs ────────────────────────────────

#[account]
pub struct Registry {
    pub authority: Pubkey,
    pub token_mint: Pubkey,
    pub total_nodes: u64,
    pub bump: u8,
}

#[account]
pub struct NodeRecord {
    pub operator: Pubkey,
    pub endpoint: String,        // max 256 bytes
    pub vram_gb: u16,
    pub gpu_count: u8,
    pub model_ids: Vec<[u8; 32]>, // up to 8 models
    pub staked_amount: u64,
    pub unstake_amount: u64,
    pub unstake_at: i64,
    pub reputation: u16,          // 0–1000
    pub jobs_completed: u64,
    pub jobs_disputed: u64,
    pub is_active: bool,
    pub registered_at: i64,
    pub bump: u8,
}

// ─────────────────────────── Contexts ───────────────────────────────────────

#[derive(Accounts)]
pub struct Initialize<'info> {
    #[account(
        init,
        payer = authority,
        space = 8 + 32 + 32 + 8 + 1,
        seeds = [b"registry"],
        bump
    )]
    pub registry: Account<'info, Registry>,
    pub token_mint: Account<'info, Mint>,
    #[account(mut)]
    pub authority: Signer<'info>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct RegisterNode<'info> {
    #[account(mut, seeds = [b"registry"], bump = registry.bump)]
    pub registry: Account<'info, Registry>,

    #[account(
        init,
        payer = operator,
        space = 8 + 32 + 4 + 256 + 2 + 1 + 4 + (8 * 32) + 8 + 8 + 8 + 2 + 8 + 8 + 1 + 8 + 1,
        seeds = [b"node", operator.key().as_ref()],
        bump
    )]
    pub node: Account<'info, NodeRecord>,

    #[account(
        init,
        payer = operator,
        associated_token::mint = token_mint,
        associated_token::authority = node,
    )]
    pub stake_token_account: Account<'info, TokenAccount>,

    #[account(mut, constraint = operator_token_account.mint == registry.token_mint)]
    pub operator_token_account: Account<'info, TokenAccount>,

    pub token_mint: Account<'info, Mint>,

    #[account(mut)]
    pub operator: Signer<'info>,
    pub token_program: Program<'info, Token>,
    pub associated_token_program: Program<'info, AssociatedToken>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct ModifyStake<'info> {
    #[account(mut, seeds = [b"node", operator.key().as_ref()], bump = node.bump)]
    pub node: Account<'info, NodeRecord>,
    #[account(mut)]
    pub stake_token_account: Account<'info, TokenAccount>,
    #[account(mut)]
    pub operator_token_account: Account<'info, TokenAccount>,
    pub operator: Signer<'info>,
    pub token_program: Program<'info, Token>,
}

#[derive(Accounts)]
pub struct BeginUnstake<'info> {
    #[account(mut, seeds = [b"node", operator.key().as_ref()], bump = node.bump)]
    pub node: Account<'info, NodeRecord>,
    pub operator: Signer<'info>,
}

#[derive(Accounts)]
pub struct WithdrawStake<'info> {
    #[account(mut, seeds = [b"node", operator.key().as_ref()], bump = node.bump)]
    pub node: Account<'info, NodeRecord>,
    #[account(mut)]
    pub stake_token_account: Account<'info, TokenAccount>,
    #[account(mut)]
    pub operator_token_account: Account<'info, TokenAccount>,
    pub operator: Signer<'info>,
    pub token_program: Program<'info, Token>,
}

#[derive(Accounts)]
pub struct RecordCompletion<'info> {
    #[account(mut)]
    pub node: Account<'info, NodeRecord>,
    // Only callable by inference-market program
    pub caller: Signer<'info>,
}

#[derive(Accounts)]
pub struct Slash<'info> {
    #[account(mut)]
    pub node: Account<'info, NodeRecord>,
    #[account(mut)]
    pub stake_token_account: Account<'info, TokenAccount>,
    #[account(mut)]
    pub treasury_token_account: Account<'info, TokenAccount>,
    /// Governance program PDA – constraint enforces only governance can slash
    pub governance_authority: Signer<'info>,
    pub token_program: Program<'info, Token>,
}

#[derive(Accounts)]
pub struct UpdateNode<'info> {
    #[account(mut, seeds = [b"node", operator.key().as_ref()], bump = node.bump)]
    pub node: Account<'info, NodeRecord>,
    pub operator: Signer<'info>,
}

// ─────────────────────────── Events ─────────────────────────────────────────

#[event]
pub struct NodeRegistered {
    pub operator: Pubkey,
    pub endpoint: String,
    pub vram_gb: u16,
    pub stake_amount: u64,
}

#[event]
pub struct NodeSlashed {
    pub operator: Pubkey,
    pub slash_amount: u64,
    pub reputation: u16,
}

// ─────────────────────────── Errors ─────────────────────────────────────────

#[error_code]
pub enum RegistryError {
    #[msg("Endpoint string too long (max 256 bytes)")]
    EndpointTooLong,
    #[msg("Cannot serve more than 8 model shards per node")]
    TooManyModels,
    #[msg("Stake amount below minimum required")]
    InsufficientStake,
    #[msg("VRAM must be greater than zero")]
    InvalidVram,
    #[msg("Amount must be greater than zero")]
    ZeroAmount,
    #[msg("Unstaking this amount would drop stake below minimum")]
    WouldDropBelowMin,
    #[msg("No pending unstake request")]
    NoPendingUnstake,
    #[msg("7-day unstake cooldown not yet met")]
    CooldownNotMet,
    #[msg("Caller is not the node operator")]
    NotOperator,
}
