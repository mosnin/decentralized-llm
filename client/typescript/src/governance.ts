/**
 * TypeScript client for the governance Anchor program.
 * Enables token holders to create proposals, vote, and track DAO decisions.
 */

import { AnchorProvider, BN, Program } from "@coral-xyz/anchor";
import { PublicKey, SystemProgram } from "@solana/web3.js";
import { TOKEN_PROGRAM_ID } from "@solana/spl-token";
import { GOVERNANCE_PROGRAM_ID } from "./sdk";

type GovernanceProgram = Program<never>;

export type VoteChoice = "for" | "against" | "abstain";

export interface Proposal {
  id: bigint;
  proposer: string;
  title: string;
  descriptionCid: string;
  proposalType: number;
  votesFor: bigint;
  votesAgainst: bigint;
  votesAbstain: bigint;
  status: "active" | "passed" | "failed" | "executed";
  votingEndsAt: Date;
  executableAt: Date | null;
}

export class GovernanceClient {
  private program: GovernanceProgram;
  private provider: AnchorProvider;

  constructor(provider: AnchorProvider, idl: object) {
    this.provider = provider;
    this.program = new Program(idl as never, GOVERNANCE_PROGRAM_ID, provider);
  }

  async getProposals(): Promise<Proposal[]> {
    const accounts = await this.program.account["proposal"].all();
    return accounts.map((a) => this._parseProposal(a.account));
  }

  async getProposal(proposalId: bigint): Promise<Proposal> {
    const [pda] = PublicKey.findProgramAddressSync(
      [Buffer.from("proposal"), new BN(proposalId.toString()).toArrayLike(Buffer, "le", 8)],
      GOVERNANCE_PROGRAM_ID
    );
    const account = await this.program.account["proposal"].fetch(pda);
    return this._parseProposal(account);
  }

  async createProposal(params: {
    title: string;
    descriptionCid: string;
    proposalType: number;
    calldata: Uint8Array;
  }): Promise<bigint> {
    const [daoPda] = PublicKey.findProgramAddressSync([Buffer.from("dao")], GOVERNANCE_PROGRAM_ID);
    const dao = await this.program.account["dao"].fetch(daoPda);
    const proposalId: BN = dao.totalProposals as BN;

    const [proposalPda] = PublicKey.findProgramAddressSync(
      [Buffer.from("proposal"), proposalId.toArrayLike(Buffer, "le", 8)],
      GOVERNANCE_PROGRAM_ID
    );

    await this.program.methods
      .createProposal(
        params.title,
        params.descriptionCid,
        params.proposalType,
        Array.from(params.calldata)
      )
      .accounts({
        dao: daoPda,
        proposal: proposalPda,
        proposer: this.provider.wallet.publicKey,
        systemProgram: SystemProgram.programId,
        tokenProgram: TOKEN_PROGRAM_ID,
      })
      .rpc();

    return BigInt(proposalId.toString());
  }

  async vote(proposalId: bigint, choice: VoteChoice): Promise<string> {
    const [proposalPda] = PublicKey.findProgramAddressSync(
      [Buffer.from("proposal"), new BN(proposalId.toString()).toArrayLike(Buffer, "le", 8)],
      GOVERNANCE_PROGRAM_ID
    );
    const [voteRecordPda] = PublicKey.findProgramAddressSync(
      [
        Buffer.from("vote"),
        proposalPda.toBuffer(),
        this.provider.wallet.publicKey.toBuffer(),
      ],
      GOVERNANCE_PROGRAM_ID
    );

    const voteArg = { [choice]: {} };

    return this.program.methods
      .castVote(voteArg)
      .accounts({
        proposal: proposalPda,
        voteRecord: voteRecordPda,
        voter: this.provider.wallet.publicKey,
        systemProgram: SystemProgram.programId,
      })
      .rpc();
  }

  async finalizeProposal(proposalId: bigint): Promise<string> {
    const [daoPda] = PublicKey.findProgramAddressSync([Buffer.from("dao")], GOVERNANCE_PROGRAM_ID);
    const [proposalPda] = PublicKey.findProgramAddressSync(
      [Buffer.from("proposal"), new BN(proposalId.toString()).toArrayLike(Buffer, "le", 8)],
      GOVERNANCE_PROGRAM_ID
    );
    return this.program.methods.finalizeProposal().accounts({ dao: daoPda, proposal: proposalPda }).rpc();
  }

  onProposalFinalized(callback: (proposalId: bigint, passed: boolean) => void): number {
    return this.program.addEventListener("ProposalFinalized", (event) => {
      const passed = "passed" in (event.status as object);
      callback(BigInt((event.proposalId as BN).toString()), passed);
    });
  }

  // ─────────────────────────── private ─────────────────────────────────────

  private _parseProposal(account: Record<string, unknown>): Proposal {
    const status = Object.keys(account["status"] as object)[0] as Proposal["status"];
    return {
      id: BigInt((account["id"] as BN).toString()),
      proposer: (account["proposer"] as PublicKey).toBase58(),
      title: account["title"] as string,
      descriptionCid: account["descriptionCid"] as string,
      proposalType: account["proposalType"] as number,
      votesFor: BigInt((account["votesFor"] as BN).toString()),
      votesAgainst: BigInt((account["votesAgainst"] as BN).toString()),
      votesAbstain: BigInt((account["votesAbstain"] as BN).toString()),
      status,
      votingEndsAt: new Date((account["votingEndsAt"] as BN).toNumber() * 1000),
      executableAt:
        (account["executableAt"] as BN).toNumber() > 0
          ? new Date((account["executableAt"] as BN).toNumber() * 1000)
          : null,
    };
  }
}
