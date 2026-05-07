use anchor_lang::prelude::*;
use anchor_spl::associated_token::AssociatedToken;
use anchor_spl::token::{self, Mint, Token, TokenAccount, Transfer};

declare_id!("5YQyZqXkJHy6V3JMxKqXyLqfP9V2A3j8Rk7mN4oD1eW");

// Protocol fee: 5% of each inference payment goes to DAO treasury
const PROTOCOL_FEE_BPS: u64 = 500;
const CHALLENGE_WINDOW_SECONDS: i64 = 300; // 5 minutes to dispute a result

#[program]
pub mod inference_market {
    use super::*;

    /// Initialize the market with a fee recipient (DAO treasury) and supported token mint.
    pub fn initialize(ctx: Context<Initialize>, treasury: Pubkey) -> Result<()> {
        let market = &mut ctx.accounts.market;
        market.authority = ctx.accounts.authority.key();
        market.treasury = treasury;
        market.token_mint = ctx.accounts.token_mint.key();
        market.total_jobs = 0;
        market.bump = ctx.bumps.market;
        Ok(())
    }

    /// Post an inference job. Tokens are locked in escrow until job is settled.
    ///
    /// `model_id`       – 32-byte identifier for the model (matches compute-registry entry)
    /// `prompt_hash`    – SHA-256 of the encrypted prompt (actual prompt delivered off-chain)
    /// `max_tokens`     – maximum output tokens (caps GPU work and payment)
    /// `payment_amount` – tokens locked for this job
    /// `deadline`       – unix timestamp; job auto-refunds if unclaimed by deadline
    pub fn post_job(
        ctx: Context<PostJob>,
        model_id: [u8; 32],
        prompt_hash: [u8; 32],
        max_tokens: u32,
        payment_amount: u64,
        deadline: i64,
    ) -> Result<()> {
        require!(payment_amount > 0, MarketError::ZeroPayment);
        require!(max_tokens > 0 && max_tokens <= 8192, MarketError::InvalidMaxTokens);
        require!(
            deadline > Clock::get()?.unix_timestamp,
            MarketError::DeadlineInPast
        );

        // Transfer tokens from client → escrow PDA
        let cpi_ctx = CpiContext::new(
            ctx.accounts.token_program.to_account_info(),
            Transfer {
                from: ctx.accounts.client_token_account.to_account_info(),
                to: ctx.accounts.escrow_token_account.to_account_info(),
                authority: ctx.accounts.client.to_account_info(),
            },
        );
        token::transfer(cpi_ctx, payment_amount)?;

        let market = &mut ctx.accounts.market;
        let job = &mut ctx.accounts.job;

        job.id = market.total_jobs;
        job.client = ctx.accounts.client.key();
        job.model_id = model_id;
        job.prompt_hash = prompt_hash;
        job.max_tokens = max_tokens;
        job.payment_amount = payment_amount;
        job.deadline = deadline;
        job.status = JobStatus::Open;
        job.node = Pubkey::default();
        job.result_hash = [0u8; 32];
        job.result_cid = String::new();
        job.claimed_at = 0;
        job.bump = ctx.bumps.job;

        market.total_jobs = market.total_jobs.checked_add(1).unwrap();

        emit!(JobPosted {
            job_id: job.id,
            client: job.client,
            model_id,
            payment_amount,
            deadline,
        });

        Ok(())
    }

    /// A registered compute node claims an open job.
    /// Node must have an active registration in the compute-registry program.
    pub fn claim_job(ctx: Context<ClaimJob>) -> Result<()> {
        let job = &mut ctx.accounts.job;

        require!(job.status == JobStatus::Open, MarketError::JobNotOpen);
        require!(
            Clock::get()?.unix_timestamp < job.deadline,
            MarketError::JobExpired
        );

        job.status = JobStatus::InProgress;
        job.node = ctx.accounts.node.key();
        job.claimed_at = Clock::get()?.unix_timestamp;

        emit!(JobClaimed {
            job_id: job.id,
            node: job.node,
        });

        Ok(())
    }

    /// Node submits the result. Result content is stored on IPFS/Arweave; only the hash goes on-chain.
    pub fn submit_result(
        ctx: Context<SubmitResult>,
        result_hash: [u8; 32],
        result_cid: String,
    ) -> Result<()> {
        require!(result_cid.len() <= 128, MarketError::CidTooLong);

        let job = &mut ctx.accounts.job;

        require!(job.status == JobStatus::InProgress, MarketError::JobNotInProgress);
        require!(job.node == ctx.accounts.node.key(), MarketError::NotJobNode);

        job.status = JobStatus::PendingAcceptance;
        job.result_hash = result_hash;
        job.result_cid = result_cid.clone();

        emit!(ResultSubmitted {
            job_id: job.id,
            node: job.node,
            result_hash,
        });

        Ok(())
    }

    /// Client accepts the result and releases payment to the node (minus protocol fee).
    pub fn accept_result(ctx: Context<AcceptResult>) -> Result<()> {
        let job = &ctx.accounts.job;

        require!(
            job.status == JobStatus::PendingAcceptance,
            MarketError::ResultNotPending
        );
        require!(job.client == ctx.accounts.client.key(), MarketError::NotJobClient);

        let protocol_fee = job
            .payment_amount
            .checked_mul(PROTOCOL_FEE_BPS)
            .unwrap()
            .checked_div(10_000)
            .unwrap();
        let node_payment = job.payment_amount.checked_sub(protocol_fee).unwrap();

        let seeds = &[b"job".as_ref(), &job.id.to_le_bytes(), &[job.bump]];
        let signer = &[&seeds[..]];

        // Pay node
        token::transfer(
            CpiContext::new_with_signer(
                ctx.accounts.token_program.to_account_info(),
                Transfer {
                    from: ctx.accounts.escrow_token_account.to_account_info(),
                    to: ctx.accounts.node_token_account.to_account_info(),
                    authority: ctx.accounts.job.to_account_info(),
                },
                signer,
            ),
            node_payment,
        )?;

        // Pay treasury
        token::transfer(
            CpiContext::new_with_signer(
                ctx.accounts.token_program.to_account_info(),
                Transfer {
                    from: ctx.accounts.escrow_token_account.to_account_info(),
                    to: ctx.accounts.treasury_token_account.to_account_info(),
                    authority: ctx.accounts.job.to_account_info(),
                },
                signer,
            ),
            protocol_fee,
        )?;

        let job = &mut ctx.accounts.job;
        job.status = JobStatus::Completed;

        emit!(JobCompleted {
            job_id: job.id,
            node: job.node,
            node_payment,
            protocol_fee,
        });

        Ok(())
    }

    /// Client disputes a result within the challenge window.
    /// Dispute resolution is handled by the DAO governance program.
    pub fn dispute_result(ctx: Context<DisputeResult>, reason: String) -> Result<()> {
        require!(reason.len() <= 256, MarketError::ReasonTooLong);

        let job = &mut ctx.accounts.job;

        require!(
            job.status == JobStatus::PendingAcceptance,
            MarketError::ResultNotPending
        );
        require!(job.client == ctx.accounts.client.key(), MarketError::NotJobClient);

        let now = Clock::get()?.unix_timestamp;
        // Auto-accept after challenge window passes (node can call this path too)
        require!(
            now <= job.claimed_at + CHALLENGE_WINDOW_SECONDS,
            MarketError::ChallengeWindowClosed
        );

        job.status = JobStatus::Disputed;

        emit!(JobDisputed {
            job_id: job.id,
            client: job.client,
            node: job.node,
            reason,
        });

        Ok(())
    }

    /// Anyone can call this to auto-settle a job after the challenge window.
    /// Used so nodes don't need the client to be responsive.
    pub fn auto_settle(ctx: Context<AutoSettle>) -> Result<()> {
        let job = &ctx.accounts.job;

        require!(
            job.status == JobStatus::PendingAcceptance,
            MarketError::ResultNotPending
        );

        let now = Clock::get()?.unix_timestamp;
        require!(
            now > job.claimed_at + CHALLENGE_WINDOW_SECONDS,
            MarketError::ChallengeWindowOpen
        );

        let protocol_fee = job
            .payment_amount
            .checked_mul(PROTOCOL_FEE_BPS)
            .unwrap()
            .checked_div(10_000)
            .unwrap();
        let node_payment = job.payment_amount.checked_sub(protocol_fee).unwrap();

        let seeds = &[b"job".as_ref(), &job.id.to_le_bytes(), &[job.bump]];
        let signer = &[&seeds[..]];

        token::transfer(
            CpiContext::new_with_signer(
                ctx.accounts.token_program.to_account_info(),
                Transfer {
                    from: ctx.accounts.escrow_token_account.to_account_info(),
                    to: ctx.accounts.node_token_account.to_account_info(),
                    authority: ctx.accounts.job.to_account_info(),
                },
                signer,
            ),
            node_payment,
        )?;

        token::transfer(
            CpiContext::new_with_signer(
                ctx.accounts.token_program.to_account_info(),
                Transfer {
                    from: ctx.accounts.escrow_token_account.to_account_info(),
                    to: ctx.accounts.treasury_token_account.to_account_info(),
                    authority: ctx.accounts.job.to_account_info(),
                },
                signer,
            ),
            protocol_fee,
        )?;

        let job = &mut ctx.accounts.job;
        job.status = JobStatus::Completed;

        emit!(JobAutoSettled { job_id: job.id });

        Ok(())
    }

    /// Refund client if job expired before any node claimed it.
    pub fn refund_expired(ctx: Context<RefundExpired>) -> Result<()> {
        let job = &ctx.accounts.job;

        require!(
            job.status == JobStatus::Open,
            MarketError::JobNotOpen
        );
        require!(
            Clock::get()?.unix_timestamp >= job.deadline,
            MarketError::JobNotExpired
        );

        let seeds = &[b"job".as_ref(), &job.id.to_le_bytes(), &[job.bump]];
        let signer = &[&seeds[..]];

        token::transfer(
            CpiContext::new_with_signer(
                ctx.accounts.token_program.to_account_info(),
                Transfer {
                    from: ctx.accounts.escrow_token_account.to_account_info(),
                    to: ctx.accounts.client_token_account.to_account_info(),
                    authority: ctx.accounts.job.to_account_info(),
                },
                signer,
            ),
            job.payment_amount,
        )?;

        let job = &mut ctx.accounts.job;
        job.status = JobStatus::Refunded;

        Ok(())
    }
}

// ─────────────────────────── Account structs ────────────────────────────────

#[account]
pub struct Market {
    pub authority: Pubkey,
    pub treasury: Pubkey,
    pub token_mint: Pubkey,
    pub total_jobs: u64,
    pub bump: u8,
}

#[account]
pub struct Job {
    pub id: u64,
    pub client: Pubkey,
    pub node: Pubkey,
    pub model_id: [u8; 32],
    pub prompt_hash: [u8; 32],
    pub result_hash: [u8; 32],
    pub result_cid: String,  // IPFS / Arweave CID (max 128 bytes)
    pub max_tokens: u32,
    pub payment_amount: u64,
    pub deadline: i64,
    pub claimed_at: i64,
    pub status: JobStatus,
    pub bump: u8,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, PartialEq, Eq)]
pub enum JobStatus {
    Open,
    InProgress,
    PendingAcceptance,
    Completed,
    Disputed,
    Refunded,
}

// ─────────────────────────── Contexts ───────────────────────────────────────

#[derive(Accounts)]
pub struct Initialize<'info> {
    #[account(
        init,
        payer = authority,
        space = 8 + 32 + 32 + 32 + 8 + 1,
        seeds = [b"market"],
        bump
    )]
    pub market: Account<'info, Market>,
    pub token_mint: Account<'info, Mint>,
    #[account(mut)]
    pub authority: Signer<'info>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
#[instruction(model_id: [u8; 32], prompt_hash: [u8; 32], max_tokens: u32, payment_amount: u64, deadline: i64)]
pub struct PostJob<'info> {
    #[account(mut, seeds = [b"market"], bump = market.bump)]
    pub market: Account<'info, Market>,

    #[account(
        init,
        payer = client,
        // 8 disc + job fields (roughly 400 bytes, padded)
        space = 8 + 8 + 32 + 32 + 32 + 32 + 32 + 4 + 128 + 4 + 8 + 8 + 8 + 1 + 1,
        seeds = [b"job", &market.total_jobs.to_le_bytes()],
        bump
    )]
    pub job: Account<'info, Job>,

    #[account(
        init,
        payer = client,
        associated_token::mint = token_mint,
        associated_token::authority = job,
    )]
    pub escrow_token_account: Account<'info, TokenAccount>,

    #[account(mut, constraint = client_token_account.mint == market.token_mint)]
    pub client_token_account: Account<'info, TokenAccount>,

    pub token_mint: Account<'info, Mint>,

    #[account(mut)]
    pub client: Signer<'info>,
    pub token_program: Program<'info, Token>,
    pub associated_token_program: Program<'info, AssociatedToken>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct ClaimJob<'info> {
    #[account(mut, seeds = [b"job", &job.id.to_le_bytes()], bump = job.bump)]
    pub job: Account<'info, Job>,
    // Node signer – verified against compute-registry via account constraint
    pub node: Signer<'info>,
}

#[derive(Accounts)]
pub struct SubmitResult<'info> {
    #[account(mut, seeds = [b"job", &job.id.to_le_bytes()], bump = job.bump)]
    pub job: Account<'info, Job>,
    pub node: Signer<'info>,
}

#[derive(Accounts)]
pub struct AcceptResult<'info> {
    #[account(mut, seeds = [b"job", &job.id.to_le_bytes()], bump = job.bump)]
    pub job: Account<'info, Job>,

    #[account(mut, constraint = escrow_token_account.mint == job.key())]
    pub escrow_token_account: Account<'info, TokenAccount>,
    #[account(mut)]
    pub node_token_account: Account<'info, TokenAccount>,
    #[account(mut)]
    pub treasury_token_account: Account<'info, TokenAccount>,

    pub client: Signer<'info>,
    pub token_program: Program<'info, Token>,
}

#[derive(Accounts)]
pub struct DisputeResult<'info> {
    #[account(mut, seeds = [b"job", &job.id.to_le_bytes()], bump = job.bump)]
    pub job: Account<'info, Job>,
    pub client: Signer<'info>,
}

#[derive(Accounts)]
pub struct AutoSettle<'info> {
    #[account(mut, seeds = [b"job", &job.id.to_le_bytes()], bump = job.bump)]
    pub job: Account<'info, Job>,
    #[account(mut)]
    pub escrow_token_account: Account<'info, TokenAccount>,
    #[account(mut)]
    pub node_token_account: Account<'info, TokenAccount>,
    #[account(mut)]
    pub treasury_token_account: Account<'info, TokenAccount>,
    pub token_program: Program<'info, Token>,
}

#[derive(Accounts)]
pub struct RefundExpired<'info> {
    #[account(mut, seeds = [b"job", &job.id.to_le_bytes()], bump = job.bump)]
    pub job: Account<'info, Job>,
    #[account(mut)]
    pub escrow_token_account: Account<'info, TokenAccount>,
    #[account(mut)]
    pub client_token_account: Account<'info, TokenAccount>,
    pub token_program: Program<'info, Token>,
}

// ─────────────────────────── Events ─────────────────────────────────────────

#[event]
pub struct JobPosted {
    pub job_id: u64,
    pub client: Pubkey,
    pub model_id: [u8; 32],
    pub payment_amount: u64,
    pub deadline: i64,
}

#[event]
pub struct JobClaimed {
    pub job_id: u64,
    pub node: Pubkey,
}

#[event]
pub struct ResultSubmitted {
    pub job_id: u64,
    pub node: Pubkey,
    pub result_hash: [u8; 32],
}

#[event]
pub struct JobCompleted {
    pub job_id: u64,
    pub node: Pubkey,
    pub node_payment: u64,
    pub protocol_fee: u64,
}

#[event]
pub struct JobDisputed {
    pub job_id: u64,
    pub client: Pubkey,
    pub node: Pubkey,
    pub reason: String,
}

#[event]
pub struct JobAutoSettled {
    pub job_id: u64,
}

// ─────────────────────────── Errors ─────────────────────────────────────────

#[error_code]
pub enum MarketError {
    #[msg("Payment amount must be greater than zero")]
    ZeroPayment,
    #[msg("max_tokens must be between 1 and 8192")]
    InvalidMaxTokens,
    #[msg("Deadline must be in the future")]
    DeadlineInPast,
    #[msg("Job is not open for claiming")]
    JobNotOpen,
    #[msg("Job has passed its deadline")]
    JobExpired,
    #[msg("Job is not in progress")]
    JobNotInProgress,
    #[msg("Caller is not the assigned node for this job")]
    NotJobNode,
    #[msg("Result is not in pending acceptance state")]
    ResultNotPending,
    #[msg("Caller is not the job client")]
    NotJobClient,
    #[msg("Challenge window has closed; use auto_settle")]
    ChallengeWindowClosed,
    #[msg("Challenge window is still open")]
    ChallengeWindowOpen,
    #[msg("Job has not yet expired")]
    JobNotExpired,
    #[msg("CID string too long (max 128 bytes)")]
    CidTooLong,
    #[msg("Dispute reason too long (max 256 bytes)")]
    ReasonTooLong,
}
