import * as anchor from "@coral-xyz/anchor";
import { Program, AnchorError } from "@coral-xyz/anchor";
import {
  PublicKey,
  Keypair,
  SystemProgram,
  LAMPORTS_PER_SOL,
} from "@solana/web3.js";
import {
  TOKEN_PROGRAM_ID,
  ASSOCIATED_TOKEN_PROGRAM_ID,
  createMint,
  createAccount,
  mintTo,
  getAccount,
  getAssociatedTokenAddress,
} from "@solana/spl-token";
import { expect } from "chai";
import { InferenceMarket } from "../target/types/inference_market";

// ─── Helpers ────────────────────────────────────────────────────────────────

async function fundKeypair(
  provider: anchor.AnchorProvider,
  kp: Keypair,
  sol = 10
): Promise<void> {
  const sig = await provider.connection.requestAirdrop(
    kp.publicKey,
    sol * LAMPORTS_PER_SOL
  );
  await provider.connection.confirmTransaction(sig);
}

function randomHash(): number[] {
  return Array.from({ length: 32 }, () => Math.floor(Math.random() * 256));
}

// ─── Suite ──────────────────────────────────────────────────────────────────

describe("inference-market", () => {
  const provider = anchor.AnchorProvider.env();
  anchor.setProvider(provider);

  const program = anchor.workspace.InferenceMarket as Program<InferenceMarket>;

  // Shared state populated during "before" hook
  let tokenMint: PublicKey;
  let authorityKp: Keypair;
  let clientKp: Keypair;
  let nodeKp: Keypair;
  let treasury: Keypair;
  let clientTokenAccount: PublicKey;
  let nodeTokenAccount: PublicKey;
  let treasuryTokenAccount: PublicKey;

  const INITIAL_SUPPLY = 1_000_000_000; // 1 billion raw units
  const PAYMENT_AMOUNT = new anchor.BN(1_000_000); // 1 token (6 decimals)

  // Seeds
  const [marketPda] = PublicKey.findProgramAddressSync(
    [Buffer.from("market")],
    program.programId
  );

  before("fund wallets and mint tokens", async () => {
    authorityKp = Keypair.generate();
    clientKp = Keypair.generate();
    nodeKp = Keypair.generate();
    treasury = Keypair.generate();

    await Promise.all([
      fundKeypair(provider, authorityKp),
      fundKeypair(provider, clientKp),
      fundKeypair(provider, nodeKp),
      fundKeypair(provider, treasury),
    ]);

    // Create SPL token mint
    tokenMint = await createMint(
      provider.connection,
      authorityKp,
      authorityKp.publicKey,
      null,
      6
    );

    // Create token accounts
    clientTokenAccount = await createAccount(
      provider.connection,
      clientKp,
      tokenMint,
      clientKp.publicKey
    );
    nodeTokenAccount = await createAccount(
      provider.connection,
      nodeKp,
      tokenMint,
      nodeKp.publicKey
    );
    treasuryTokenAccount = await createAccount(
      provider.connection,
      treasury,
      tokenMint,
      treasury.publicKey
    );

    // Mint tokens to client
    await mintTo(
      provider.connection,
      authorityKp,
      tokenMint,
      clientTokenAccount,
      authorityKp,
      INITIAL_SUPPLY
    );
  });

  // ── Test 1: Initialize market ──────────────────────────────────────────────
  it("initializes the market with treasury and token mint", async () => {
    await program.methods
      .initialize(treasury.publicKey)
      .accounts({
        market: marketPda,
        tokenMint,
        authority: authorityKp.publicKey,
        systemProgram: SystemProgram.programId,
      })
      .signers([authorityKp])
      .rpc();

    const market = await program.account.market.fetch(marketPda);
    expect(market.authority.toBase58()).to.equal(
      authorityKp.publicKey.toBase58()
    );
    expect(market.treasury.toBase58()).to.equal(treasury.publicKey.toBase58());
    expect(market.tokenMint.toBase58()).to.equal(tokenMint.toBase58());
    expect(market.totalJobs.toNumber()).to.equal(0);
  });

  // ── Test 2: Post a job (creates escrow and locks tokens) ──────────────────
  it("posts a job, creates escrow PDA, and locks tokens", async () => {
    const marketBefore = await program.account.market.fetch(marketPda);
    const jobId = marketBefore.totalJobs; // BN

    const [jobPda] = PublicKey.findProgramAddressSync(
      [Buffer.from("job"), jobId.toArrayLike(Buffer, "le", 8)],
      program.programId
    );

    const escrowTokenAccount = await getAssociatedTokenAddress(
      tokenMint,
      jobPda,
      true // allow off-curve PDA owner
    );

    const modelId = randomHash();
    const promptHash = randomHash();
    const promptCid = "QmTestCidForPrompt1234567890";
    const maxTokens = 512;
    const deadline = new anchor.BN(Math.floor(Date.now() / 1000) + 3600); // +1 hour

    const clientBalanceBefore = (
      await getAccount(provider.connection, clientTokenAccount)
    ).amount;

    await program.methods
      .postJob(
        modelId,
        promptHash,
        promptCid,
        maxTokens,
        PAYMENT_AMOUNT,
        deadline
      )
      .accounts({
        market: marketPda,
        job: jobPda,
        escrowTokenAccount,
        clientTokenAccount,
        tokenMint,
        client: clientKp.publicKey,
        tokenProgram: TOKEN_PROGRAM_ID,
        associatedTokenProgram: ASSOCIATED_TOKEN_PROGRAM_ID,
        systemProgram: SystemProgram.programId,
      })
      .signers([clientKp])
      .rpc();

    // Verify job account state
    const job = await program.account.job.fetch(jobPda);
    expect(job.id.toNumber()).to.equal(jobId.toNumber());
    expect(job.client.toBase58()).to.equal(clientKp.publicKey.toBase58());
    expect(job.paymentAmount.toNumber()).to.equal(PAYMENT_AMOUNT.toNumber());
    expect(job.maxTokens).to.equal(maxTokens);
    expect(JSON.stringify(Object.keys(job.status))).to.include("open");

    // Verify tokens moved from client → escrow
    const escrowBalance = (
      await getAccount(provider.connection, escrowTokenAccount)
    ).amount;
    expect(escrowBalance.toString()).to.equal(
      PAYMENT_AMOUNT.toString()
    );

    const clientBalanceAfter = (
      await getAccount(provider.connection, clientTokenAccount)
    ).amount;
    expect(
      (clientBalanceBefore - clientBalanceAfter).toString()
    ).to.equal(PAYMENT_AMOUNT.toString());

    // Verify market counter incremented
    const marketAfter = await program.account.market.fetch(marketPda);
    expect(marketAfter.totalJobs.toNumber()).to.equal(jobId.toNumber() + 1);
  });

  // ── Test 3: Claim a job ────────────────────────────────────────────────────
  it("allows a compute node to claim an open job", async () => {
    // The first job (id=0) was created in the previous test
    const jobId = new anchor.BN(0);
    const [jobPda] = PublicKey.findProgramAddressSync(
      [Buffer.from("job"), jobId.toArrayLike(Buffer, "le", 8)],
      program.programId
    );

    await program.methods
      .claimJob()
      .accounts({
        job: jobPda,
        node: nodeKp.publicKey,
      })
      .signers([nodeKp])
      .rpc();

    const job = await program.account.job.fetch(jobPda);
    expect(JSON.stringify(Object.keys(job.status))).to.include("inProgress");
    expect(job.node.toBase58()).to.equal(nodeKp.publicKey.toBase58());
    expect(job.claimedAt.toNumber()).to.be.greaterThan(0);
  });

  // ── Test 4: Submit a result hash ──────────────────────────────────────────
  it("allows the assigned node to submit a result hash", async () => {
    const jobId = new anchor.BN(0);
    const [jobPda] = PublicKey.findProgramAddressSync(
      [Buffer.from("job"), jobId.toArrayLike(Buffer, "le", 8)],
      program.programId
    );

    const resultHash = randomHash();
    const resultCid = "QmResultCidForJob0ABCDEFGHIJ";

    await program.methods
      .submitResult(resultHash, resultCid)
      .accounts({
        job: jobPda,
        node: nodeKp.publicKey,
      })
      .signers([nodeKp])
      .rpc();

    const job = await program.account.job.fetch(jobPda);
    expect(
      JSON.stringify(Object.keys(job.status))
    ).to.include("pendingAcceptance");
    expect(job.resultCid).to.equal(resultCid);
    expect(Buffer.from(job.resultHash).toString("hex")).to.equal(
      Buffer.from(resultHash).toString("hex")
    );
  });

  // ── Test 5: Auto-settle rejects early settlement ──────────────────────────
  it("rejects auto_settle when the 5-minute challenge window is still open", async () => {
    // Job 0 was just claimed seconds ago — challenge window (300 s) is open
    const jobId = new anchor.BN(0);
    const [jobPda] = PublicKey.findProgramAddressSync(
      [Buffer.from("job"), jobId.toArrayLike(Buffer, "le", 8)],
      program.programId
    );

    const escrowTokenAccount = await getAssociatedTokenAddress(
      tokenMint,
      jobPda,
      true
    );

    let threw = false;
    try {
      await program.methods
        .autoSettle()
        .accounts({
          job: jobPda,
          escrowTokenAccount,
          nodeTokenAccount,
          treasuryTokenAccount,
          tokenProgram: TOKEN_PROGRAM_ID,
        })
        .rpc();
    } catch (err: unknown) {
      threw = true;
      // Anchor wraps program errors; verify it is the ChallengeWindowOpen error
      if (err instanceof AnchorError) {
        expect(err.error.errorCode.code).to.equal("ChallengeWindowOpen");
      }
    }
    expect(threw).to.be.true;
  });

  // ── Test 6: Client can dispute within challenge window ────────────────────
  it("allows the client to dispute a result within the challenge window", async () => {
    const jobId = new anchor.BN(0);
    const [jobPda] = PublicKey.findProgramAddressSync(
      [Buffer.from("job"), jobId.toArrayLike(Buffer, "le", 8)],
      program.programId
    );

    await program.methods
      .disputeResult("Output was truncated and incoherent")
      .accounts({
        job: jobPda,
        client: clientKp.publicKey,
      })
      .signers([clientKp])
      .rpc();

    const job = await program.account.job.fetch(jobPda);
    expect(JSON.stringify(Object.keys(job.status))).to.include("disputed");
  });

  // ── Test 7: post_job rejects zero payment ─────────────────────────────────
  it("rejects posting a job with zero payment", async () => {
    const marketState = await program.account.market.fetch(marketPda);
    const jobId = marketState.totalJobs;

    const [jobPda] = PublicKey.findProgramAddressSync(
      [Buffer.from("job"), jobId.toArrayLike(Buffer, "le", 8)],
      program.programId
    );

    const escrowTokenAccount = await getAssociatedTokenAddress(
      tokenMint,
      jobPda,
      true
    );

    let threw = false;
    try {
      await program.methods
        .postJob(
          randomHash(),
          randomHash(),
          "QmValidCid",
          512,
          new anchor.BN(0), // zero payment — should fail
          new anchor.BN(Math.floor(Date.now() / 1000) + 3600)
        )
        .accounts({
          market: marketPda,
          job: jobPda,
          escrowTokenAccount,
          clientTokenAccount,
          tokenMint,
          client: clientKp.publicKey,
          tokenProgram: TOKEN_PROGRAM_ID,
          associatedTokenProgram: ASSOCIATED_TOKEN_PROGRAM_ID,
          systemProgram: SystemProgram.programId,
        })
        .signers([clientKp])
        .rpc();
    } catch (err: unknown) {
      threw = true;
      if (err instanceof AnchorError) {
        expect(err.error.errorCode.code).to.equal("ZeroPayment");
      }
    }
    expect(threw).to.be.true;
  });

  // ── Test 8: post_job rejects deadline in the past ─────────────────────────
  it("rejects posting a job with a deadline in the past", async () => {
    const marketState = await program.account.market.fetch(marketPda);
    const jobId = marketState.totalJobs;

    const [jobPda] = PublicKey.findProgramAddressSync(
      [Buffer.from("job"), jobId.toArrayLike(Buffer, "le", 8)],
      program.programId
    );

    const escrowTokenAccount = await getAssociatedTokenAddress(
      tokenMint,
      jobPda,
      true
    );

    let threw = false;
    try {
      await program.methods
        .postJob(
          randomHash(),
          randomHash(),
          "QmValidCid",
          512,
          PAYMENT_AMOUNT,
          new anchor.BN(Math.floor(Date.now() / 1000) - 100) // past deadline
        )
        .accounts({
          market: marketPda,
          job: jobPda,
          escrowTokenAccount,
          clientTokenAccount,
          tokenMint,
          client: clientKp.publicKey,
          tokenProgram: TOKEN_PROGRAM_ID,
          associatedTokenProgram: ASSOCIATED_TOKEN_PROGRAM_ID,
          systemProgram: SystemProgram.programId,
        })
        .signers([clientKp])
        .rpc();
    } catch (err: unknown) {
      threw = true;
      if (err instanceof AnchorError) {
        expect(err.error.errorCode.code).to.equal("DeadlineInPast");
      }
    }
    expect(threw).to.be.true;
  });

  // ── Test 9: Wrong node cannot submit result ────────────────────────────────
  it("rejects result submission from a node that did not claim the job", async () => {
    // Post a fresh job so we have one in InProgress with a known node
    const marketState = await program.account.market.fetch(marketPda);
    const jobId = marketState.totalJobs;

    const [jobPda] = PublicKey.findProgramAddressSync(
      [Buffer.from("job"), jobId.toArrayLike(Buffer, "le", 8)],
      program.programId
    );
    const escrowTokenAccount = await getAssociatedTokenAddress(
      tokenMint,
      jobPda,
      true
    );

    await program.methods
      .postJob(
        randomHash(),
        randomHash(),
        "QmFreshJobCid",
        128,
        PAYMENT_AMOUNT,
        new anchor.BN(Math.floor(Date.now() / 1000) + 3600)
      )
      .accounts({
        market: marketPda,
        job: jobPda,
        escrowTokenAccount,
        clientTokenAccount,
        tokenMint,
        client: clientKp.publicKey,
        tokenProgram: TOKEN_PROGRAM_ID,
        associatedTokenProgram: ASSOCIATED_TOKEN_PROGRAM_ID,
        systemProgram: SystemProgram.programId,
      })
      .signers([clientKp])
      .rpc();

    // Node claims the job
    await program.methods
      .claimJob()
      .accounts({ job: jobPda, node: nodeKp.publicKey })
      .signers([nodeKp])
      .rpc();

    // Imposter tries to submit result
    const imposterKp = Keypair.generate();
    await fundKeypair(provider, imposterKp, 1);

    let threw = false;
    try {
      await program.methods
        .submitResult(randomHash(), "QmFakeResult")
        .accounts({ job: jobPda, node: imposterKp.publicKey })
        .signers([imposterKp])
        .rpc();
    } catch (err: unknown) {
      threw = true;
      if (err instanceof AnchorError) {
        expect(err.error.errorCode.code).to.equal("NotJobNode");
      }
    }
    expect(threw).to.be.true;
  });
});
