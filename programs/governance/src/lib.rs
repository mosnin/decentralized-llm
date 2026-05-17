use anchor_lang::prelude::*;
use anchor_spl::token::{self, Token, TokenAccount};

declare_id!("3CvE7tX9rMwPfBgY2nKjH6oL4sQ8uZaD5mR1iW0eN9T");

/// Quorum: 4% of total supply must vote for a proposal to pass
const QUORUM_BPS: u64 = 400;
/// Super-majority threshold for passing (60%)
const PASS_THRESHOLD_BPS: u64 = 6_000;
/// Voting period: 7 days
const VOTING_PERIOD_SECONDS: i64 = 7 * 24 * 3600;
/// Time-lock before execution: 48 hours
const EXECUTION_DELAY_SECONDS: i64 = 48 * 3600;
/// Proposal bond (burned on failed proposals to prevent spam): 1000 tokens
const PROPOSAL_BOND: u64 = 1_000 * 1_000_000;

#[program]
pub mod governance {
    use super::*;

    pub fn initialize(ctx: Context<Initialize>, total_supply: u64) -> Result<()> {
        let dao = &mut ctx.accounts.dao;
        dao.authority = ctx.accounts.authority.key();
        dao.token_mint = ctx.accounts.token_mint.key();
        dao.total_supply = total_supply;
        dao.total_proposals = 0;
        dao.bump = ctx.bumps.dao;
        Ok(())
    }

    /// Create a governance proposal.
    ///
    /// Proposal types:
    ///   0 - Deploy new model (model_id in calldata)
    ///   1 - Retire model
    ///   2 - Update protocol fee
    ///   3 - Update min stake
    ///   4 - Grant emergency pause
    ///   5 - Treasury spend
    ///   6 - Parameter change (freeform)
    ///
    /// Proposer locks `PROPOSAL_BOND` tokens; returned on proposal success.
    pub fn create_proposal(
        ctx: Context<CreateProposal>,
        title: String,
        description_cid: String, // IPFS CID of full description
        proposal_type: u8,
        calldata: Vec<u8>,
    ) -> Result<()> {
        require!(title.len() <= 128, GovError::TitleTooLong);
        require!(description_cid.len() <= 128, GovError::CidTooLong);
        require!(proposal_type <= 6, GovError::InvalidProposalType);
        require!(calldata.len() <= 512, GovError::CalldataTooLong);

        // Lock proposal bond
        token::transfer(
            CpiContext::new(
                ctx.accounts.token_program.to_account_info(),
                token::Transfer {
                    from: ctx.accounts.proposer_token_account.to_account_info(),
                    to: ctx.accounts.bond_token_account.to_account_info(),
                    authority: ctx.accounts.proposer.to_account_info(),
                },
            ),
            PROPOSAL_BOND,
        )?;

        let dao = &mut ctx.accounts.dao;
        let proposal = &mut ctx.accounts.proposal;
        let now = Clock::get()?.unix_timestamp;

        proposal.id = dao.total_proposals;
        proposal.proposer = ctx.accounts.proposer.key();
        proposal.title = title;
        proposal.description_cid = description_cid;
        proposal.proposal_type = proposal_type;
        proposal.calldata = calldata;
        proposal.votes_for = 0;
        proposal.votes_against = 0;
        proposal.votes_abstain = 0;
        proposal.created_at = now;
        proposal.voting_ends_at = now + VOTING_PERIOD_SECONDS;
        proposal.executable_at = 0;
        proposal.status = ProposalStatus::Active;
        proposal.bump = ctx.bumps.proposal;

        dao.total_proposals = dao.total_proposals.checked_add(1).unwrap();

        emit!(ProposalCreated {
            proposal_id: proposal.id,
            proposer: proposal.proposer,
            proposal_type,
        });

        Ok(())
    }

    /// Cast a vote. Voting power = token balance at vote time (snapshot-less for simplicity;
    /// a production system would use a snapshot mechanism to prevent vote buying).
    pub fn cast_vote(
        ctx: Context<CastVote>,
        vote: VoteChoice,
    ) -> Result<()> {
        let proposal = &ctx.accounts.proposal;

        require!(proposal.status == ProposalStatus::Active, GovError::ProposalNotActive);
        require!(
            Clock::get()?.unix_timestamp <= proposal.voting_ends_at,
            GovError::VotingEnded
        );

        let voting_power = ctx.accounts.voter_token_account.amount;
        require!(voting_power > 0, GovError::NoVotingPower);

        let vote_record = &mut ctx.accounts.vote_record;
        // init-if-needed handles first-time creation; re-vote is rejected by the check below
        require!(!vote_record.has_voted, GovError::AlreadyVoted);

        vote_record.voter = ctx.accounts.voter.key();
        vote_record.proposal_id = proposal.id;
        vote_record.choice = vote.clone();
        vote_record.voting_power = voting_power;
        vote_record.has_voted = true;

        let proposal = &mut ctx.accounts.proposal;
        match vote {
            VoteChoice::For     => proposal.votes_for += voting_power,
            VoteChoice::Against => proposal.votes_against += voting_power,
            VoteChoice::Abstain => proposal.votes_abstain += voting_power,
        }

        emit!(VoteCast {
            proposal_id: proposal.id,
            voter: ctx.accounts.voter.key(),
            choice: vote_record.choice.clone(),
            voting_power,
        });

        Ok(())
    }

    /// Finalize voting after the voting period ends.
    /// Anyone can call this — it's permissionless so proposals don't stall.
    pub fn finalize_proposal(ctx: Context<FinalizeProposal>) -> Result<()> {
        let proposal = &ctx.accounts.proposal;
        let dao = &ctx.accounts.dao;

        require!(proposal.status == ProposalStatus::Active, GovError::ProposalNotActive);
        require!(
            Clock::get()?.unix_timestamp > proposal.voting_ends_at,
            GovError::VotingNotEnded
        );

        let total_votes = proposal
            .votes_for
            .checked_add(proposal.votes_against)
            .unwrap()
            .checked_add(proposal.votes_abstain)
            .unwrap();

        let quorum = dao
            .total_supply
            .checked_mul(QUORUM_BPS)
            .unwrap()
            .checked_div(10_000)
            .unwrap();

        let proposal = &mut ctx.accounts.proposal;

        if total_votes < quorum {
            proposal.status = ProposalStatus::Failed;
        } else {
            let pass_threshold = total_votes
                .checked_mul(PASS_THRESHOLD_BPS)
                .unwrap()
                .checked_div(10_000)
                .unwrap();

            if proposal.votes_for >= pass_threshold {
                proposal.status = ProposalStatus::Passed;
                proposal.executable_at =
                    Clock::get()?.unix_timestamp + EXECUTION_DELAY_SECONDS;
            } else {
                proposal.status = ProposalStatus::Failed;
            }
        }

        emit!(ProposalFinalized {
            proposal_id: proposal.id,
            status: proposal.status.clone(),
            votes_for: proposal.votes_for,
            votes_against: proposal.votes_against,
        });

        Ok(())
    }

    /// Execute a passed proposal after the time-lock delay.
    /// The actual execution effect is emitted as an event; off-chain multisig
    /// or program upgrade authority acts on the instruction payload.
    pub fn execute_proposal(ctx: Context<ExecuteProposal>) -> Result<()> {
        let proposal = &ctx.accounts.proposal;

        require!(proposal.status == ProposalStatus::Passed, GovError::ProposalNotPassed);
        require!(
            Clock::get()?.unix_timestamp >= proposal.executable_at,
            GovError::TimelockNotMet
        );

        // Return bond to proposer on successful proposal
        let seeds = &[b"proposal".as_ref(), &proposal.id.to_le_bytes(), &[proposal.bump]];
        let signer = &[&seeds[..]];

        token::transfer(
            CpiContext::new_with_signer(
                ctx.accounts.token_program.to_account_info(),
                token::Transfer {
                    from: ctx.accounts.bond_token_account.to_account_info(),
                    to: ctx.accounts.proposer_token_account.to_account_info(),
                    authority: ctx.accounts.proposal.to_account_info(),
                },
                signer,
            ),
            PROPOSAL_BOND,
        )?;

        let proposal = &mut ctx.accounts.proposal;
        proposal.status = ProposalStatus::Executed;

        emit!(ProposalExecuted {
            proposal_id: proposal.id,
            proposal_type: proposal.proposal_type,
            calldata: proposal.calldata.clone(),
        });

        Ok(())
    }
}

// ─────────────────────────── Account structs ────────────────────────────────

#[account]
pub struct Dao {
    pub authority: Pubkey,
    pub token_mint: Pubkey,
    pub total_supply: u64,
    pub total_proposals: u64,
    pub bump: u8,
}

#[account]
pub struct Proposal {
    pub id: u64,
    pub proposer: Pubkey,
    pub title: String,           // max 128
    pub description_cid: String, // max 128
    pub proposal_type: u8,
    pub calldata: Vec<u8>,       // max 512
    pub votes_for: u64,
    pub votes_against: u64,
    pub votes_abstain: u64,
    pub created_at: i64,
    pub voting_ends_at: i64,
    pub executable_at: i64,
    pub status: ProposalStatus,
    pub bump: u8,
}

#[account]
pub struct VoteRecord {
    pub voter: Pubkey,
    pub proposal_id: u64,
    pub choice: VoteChoice,
    pub voting_power: u64,
    pub has_voted: bool,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, PartialEq, Eq)]
pub enum ProposalStatus {
    Active,
    Passed,
    Failed,
    Executed,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, PartialEq, Eq)]
pub enum VoteChoice {
    For,
    Against,
    Abstain,
}

// ─────────────────────────── Contexts ───────────────────────────────────────

#[derive(Accounts)]
pub struct Initialize<'info> {
    #[account(
        init,
        payer = authority,
        space = 8 + 32 + 32 + 8 + 8 + 1,
        seeds = [b"dao"],
        bump
    )]
    pub dao: Account<'info, Dao>,
    /// CHECK: just stores the pubkey
    pub token_mint: AccountInfo<'info>,
    #[account(mut)]
    pub authority: Signer<'info>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct CreateProposal<'info> {
    #[account(mut, seeds = [b"dao"], bump = dao.bump)]
    pub dao: Account<'info, Dao>,

    #[account(
        init,
        payer = proposer,
        space = 8 + 8 + 32 + 4 + 128 + 4 + 128 + 1 + 4 + 512 + 8 + 8 + 8 + 8 + 8 + 8 + 1 + 1,
        seeds = [b"proposal", &dao.total_proposals.to_le_bytes()],
        bump
    )]
    pub proposal: Account<'info, Proposal>,

    #[account(mut)]
    pub bond_token_account: Account<'info, TokenAccount>,
    #[account(mut)]
    pub proposer_token_account: Account<'info, TokenAccount>,

    #[account(mut)]
    pub proposer: Signer<'info>,
    pub token_program: Program<'info, Token>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct CastVote<'info> {
    #[account(mut, seeds = [b"proposal", &proposal.id.to_le_bytes()], bump = proposal.bump)]
    pub proposal: Account<'info, Proposal>,

    #[account(
        init_if_needed,
        payer = voter,
        space = 8 + 32 + 8 + 1 + 8 + 1,
        seeds = [b"vote", proposal.key().as_ref(), voter.key().as_ref()],
        bump
    )]
    pub vote_record: Account<'info, VoteRecord>,

    pub voter_token_account: Account<'info, TokenAccount>,

    #[account(mut)]
    pub voter: Signer<'info>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct FinalizeProposal<'info> {
    #[account(seeds = [b"dao"], bump = dao.bump)]
    pub dao: Account<'info, Dao>,
    #[account(mut, seeds = [b"proposal", &proposal.id.to_le_bytes()], bump = proposal.bump)]
    pub proposal: Account<'info, Proposal>,
}

#[derive(Accounts)]
pub struct ExecuteProposal<'info> {
    #[account(mut, seeds = [b"proposal", &proposal.id.to_le_bytes()], bump = proposal.bump)]
    pub proposal: Account<'info, Proposal>,

    #[account(mut)]
    pub bond_token_account: Account<'info, TokenAccount>,
    #[account(mut)]
    pub proposer_token_account: Account<'info, TokenAccount>,

    pub token_program: Program<'info, Token>,
}

// ─────────────────────────── Events ─────────────────────────────────────────

#[event]
pub struct ProposalCreated {
    pub proposal_id: u64,
    pub proposer: Pubkey,
    pub proposal_type: u8,
}

#[event]
pub struct VoteCast {
    pub proposal_id: u64,
    pub voter: Pubkey,
    pub choice: VoteChoice,
    pub voting_power: u64,
}

#[event]
pub struct ProposalFinalized {
    pub proposal_id: u64,
    pub status: ProposalStatus,
    pub votes_for: u64,
    pub votes_against: u64,
}

#[event]
pub struct ProposalExecuted {
    pub proposal_id: u64,
    pub proposal_type: u8,
    pub calldata: Vec<u8>,
}

// ─────────────────────────── Errors ─────────────────────────────────────────

#[error_code]
pub enum GovError {
    #[msg("Title too long (max 128 bytes)")]
    TitleTooLong,
    #[msg("CID too long (max 128 bytes)")]
    CidTooLong,
    #[msg("Invalid proposal type (0–6)")]
    InvalidProposalType,
    #[msg("Calldata too long (max 512 bytes)")]
    CalldataTooLong,
    #[msg("Proposal is not active")]
    ProposalNotActive,
    #[msg("Voting period has ended")]
    VotingEnded,
    #[msg("No token balance; no voting power")]
    NoVotingPower,
    #[msg("You have already voted on this proposal")]
    AlreadyVoted,
    #[msg("Voting period has not ended yet")]
    VotingNotEnded,
    #[msg("Proposal did not pass")]
    ProposalNotPassed,
    #[msg("Time-lock delay not yet met (48h after passing)")]
    TimelockNotMet,
}
