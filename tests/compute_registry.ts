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
import { ComputeRegistry } from "../target/types/compute_registry";

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

function randomModelId(): number[] {
  return Array.from({ length: 32 }, () => Math.floor(Math.random() * 256));
}

// ─── Suite ──────────────────────────────────────────────────────────────────

describe("compute-registry", () => {
  const provider = anchor.AnchorProvider.env();
  anchor.setProvider(provider);

  const program = anchor.workspace.ComputeRegistry as Program<ComputeRegistry>;

  // Shared state
  let tokenMint: PublicKey;
  let authorityKp: Keypair;
  let operatorKp: Keypair;
  let govKp: Keypair; // stands in for the governance authority
  let treasuryKp: Keypair;
  let operatorTokenAccount: PublicKey;
  let treasuryTokenAccount: PublicKey;

  // 100 000 tokens at 6 decimals = minimum stake
  const MIN_STAKE = new anchor.BN(100_000 * 1_000_000);
  const INITIAL_BALANCE = new anchor.BN(10_000_000 * 1_000_000); // 10 M tokens

  const [registryPda] = PublicKey.findProgramAddressSync(
    [Buffer.from("registry")],
    program.programId
  );

  before("fund wallets and mint tokens", async () => {
    authorityKp = Keypair.generate();
    operatorKp = Keypair.generate();
    govKp = Keypair.generate();
    treasuryKp = Keypair.generate();

    await Promise.all([
      fundKeypair(provider, authorityKp),
      fundKeypair(provider, operatorKp),
      fundKeypair(provider, govKp),
      fundKeypair(provider, treasuryKp),
    ]);

    tokenMint = await createMint(
      provider.connection,
      authorityKp,
      authorityKp.publicKey,
      null,
      6
    );

    operatorTokenAccount = await createAccount(
      provider.connection,
      operatorKp,
      tokenMint,
      operatorKp.publicKey
    );

    treasuryTokenAccount = await createAccount(
      provider.connection,
      treasuryKp,
      tokenMint,
      treasuryKp.publicKey
    );

    await mintTo(
      provider.connection,
      authorityKp,
      tokenMint,
      operatorTokenAccount,
      authorityKp,
      INITIAL_BALANCE.toNumber()
    );
  });

  // ── Test 1: Initialize registry ───────────────────────────────────────────
  it("initializes the compute registry", async () => {
    await program.methods
      .initialize()
      .accounts({
        registry: registryPda,
        tokenMint,
        authority: authorityKp.publicKey,
        systemProgram: SystemProgram.programId,
      })
      .signers([authorityKp])
      .rpc();

    const registry = await program.account.registry.fetch(registryPda);
    expect(registry.authority.toBase58()).to.equal(
      authorityKp.publicKey.toBase58()
    );
    expect(registry.tokenMint.toBase58()).to.equal(tokenMint.toBase58());
    expect(registry.totalNodes.toNumber()).to.equal(0);
  });

  // ── Test 2: Register a node with minimum stake ────────────────────────────
  it("registers a compute node and locks the minimum stake", async () => {
    const [nodePda] = PublicKey.findProgramAddressSync(
      [Buffer.from("node"), operatorKp.publicKey.toBuffer()],
      program.programId
    );

    const stakeTokenAccount = await getAssociatedTokenAddress(
      tokenMint,
      nodePda,
      true
    );

    const operatorBalanceBefore = (
      await getAccount(provider.connection, operatorTokenAccount)
    ).amount;

    const modelIds = [randomModelId(), randomModelId()];

    await program.methods
      .registerNode(
        "/ip4/127.0.0.1/tcp/4001",
        80, // vram_gb
        4,  // gpu_count
        modelIds,
        MIN_STAKE
      )
      .accounts({
        registry: registryPda,
        node: nodePda,
        stakeTokenAccount,
        operatorTokenAccount,
        tokenMint,
        operator: operatorKp.publicKey,
        tokenProgram: TOKEN_PROGRAM_ID,
        associatedTokenProgram: ASSOCIATED_TOKEN_PROGRAM_ID,
        systemProgram: SystemProgram.programId,
      })
      .signers([operatorKp])
      .rpc();

    const node = await program.account.nodeRecord.fetch(nodePda);
    expect(node.operator.toBase58()).to.equal(operatorKp.publicKey.toBase58());
    expect(node.endpoint).to.equal("/ip4/127.0.0.1/tcp/4001");
    expect(node.vramGb).to.equal(80);
    expect(node.gpuCount).to.equal(4);
    expect(node.stakedAmount.toString()).to.equal(MIN_STAKE.toString());
    expect(node.reputation).to.equal(1000);
    expect(node.isActive).to.be.true;
    expect(node.jobsCompleted.toNumber()).to.equal(0);

    // Verify tokens moved into the stake escrow
    const stakeBalance = (
      await getAccount(provider.connection, stakeTokenAccount)
    ).amount;
    expect(stakeBalance.toString()).to.equal(MIN_STAKE.toString());

    const operatorBalanceAfter = (
      await getAccount(provider.connection, operatorTokenAccount)
    ).amount;
    expect(
      (operatorBalanceBefore - operatorBalanceAfter).toString()
    ).to.equal(MIN_STAKE.toString());

    // Registry counter incremented
    const registry = await program.account.registry.fetch(registryPda);
    expect(registry.totalNodes.toNumber()).to.equal(1);
  });

  // ── Test 3: Update node endpoint (heartbeat) ──────────────────────────────
  it("allows the operator to update the node endpoint", async () => {
    const [nodePda] = PublicKey.findProgramAddressSync(
      [Buffer.from("node"), operatorKp.publicKey.toBuffer()],
      program.programId
    );

    const newEndpoint = "/ip4/203.0.113.42/tcp/4001/p2p/QmNewPeerId";

    await program.methods
      .updateEndpoint(newEndpoint)
      .accounts({
        node: nodePda,
        operator: operatorKp.publicKey,
      })
      .signers([operatorKp])
      .rpc();

    const node = await program.account.nodeRecord.fetch(nodePda);
    expect(node.endpoint).to.equal(newEndpoint);
  });

  // ── Test 4: Rejects registration with insufficient stake ──────────────────
  it("rejects node registration with stake below the minimum", async () => {
    const underfundedKp = Keypair.generate();
    await fundKeypair(provider, underfundedKp, 5);

    const underfundedTokenAccount = await createAccount(
      provider.connection,
      underfundedKp,
      tokenMint,
      underfundedKp.publicKey
    );

    // Mint just under the minimum (50k tokens instead of 100k)
    await mintTo(
      provider.connection,
      authorityKp,
      tokenMint,
      underfundedTokenAccount,
      authorityKp,
      50_000 * 1_000_000
    );

    const [nodePda] = PublicKey.findProgramAddressSync(
      [Buffer.from("node"), underfundedKp.publicKey.toBuffer()],
      program.programId
    );

    const stakeTokenAccount = await getAssociatedTokenAddress(
      tokenMint,
      nodePda,
      true
    );

    const belowMinStake = new anchor.BN(50_000 * 1_000_000);

    let threw = false;
    try {
      await program.methods
        .registerNode(
          "http://localhost:8080",
          24,
          1,
          [randomModelId()],
          belowMinStake
        )
        .accounts({
          registry: registryPda,
          node: nodePda,
          stakeTokenAccount,
          operatorTokenAccount: underfundedTokenAccount,
          tokenMint,
          operator: underfundedKp.publicKey,
          tokenProgram: TOKEN_PROGRAM_ID,
          associatedTokenProgram: ASSOCIATED_TOKEN_PROGRAM_ID,
          systemProgram: SystemProgram.programId,
        })
        .signers([underfundedKp])
        .rpc();
    } catch (err: unknown) {
      threw = true;
      if (err instanceof AnchorError) {
        expect(err.error.errorCode.code).to.equal("InsufficientStake");
      }
    }
    expect(threw).to.be.true;
  });

  // ── Test 5: Slash a misbehaving node ──────────────────────────────────────
  it("slashes 10% of stake and reduces reputation by 100 on confirmed misbehavior", async () => {
    const [nodePda] = PublicKey.findProgramAddressSync(
      [Buffer.from("node"), operatorKp.publicKey.toBuffer()],
      program.programId
    );

    const stakeTokenAccount = await getAssociatedTokenAddress(
      tokenMint,
      nodePda,
      true
    );

    const nodeBefore = await program.account.nodeRecord.fetch(nodePda);
    const stakedBefore = nodeBefore.stakedAmount.toNumber();
    const reputationBefore = nodeBefore.reputation;

    // governance authority signs the slash
    await program.methods
      .slash()
      .accounts({
        node: nodePda,
        stakeTokenAccount,
        treasuryTokenAccount,
        governanceAuthority: govKp.publicKey,
        tokenProgram: TOKEN_PROGRAM_ID,
      })
      .signers([govKp])
      .rpc();

    const nodeAfter = await program.account.nodeRecord.fetch(nodePda);
    const expectedSlash = Math.floor((stakedBefore * 1000) / 10_000); // 10%
    const expectedStakeAfter = stakedBefore - expectedSlash;

    expect(nodeAfter.stakedAmount.toNumber()).to.equal(expectedStakeAfter);
    expect(nodeAfter.reputation).to.equal(reputationBefore - 100);
    expect(nodeAfter.jobsDisputed.toNumber()).to.equal(1);

    // Treasury received the slashed tokens
    const treasuryBalance = (
      await getAccount(provider.connection, treasuryTokenAccount)
    ).amount;
    expect(Number(treasuryBalance)).to.equal(expectedSlash);
  });

  // ── Test 6: Only the operator can update endpoint ─────────────────────────
  it("rejects endpoint update from a non-operator account", async () => {
    const [nodePda] = PublicKey.findProgramAddressSync(
      [Buffer.from("node"), operatorKp.publicKey.toBuffer()],
      program.programId
    );

    const intruderKp = Keypair.generate();
    await fundKeypair(provider, intruderKp, 1);

    let threw = false;
    try {
      await program.methods
        .updateEndpoint("http://malicious.host/rpc")
        .accounts({
          node: nodePda,
          operator: intruderKp.publicKey,
        })
        .signers([intruderKp])
        .rpc();
    } catch (err: unknown) {
      threw = true;
      if (err instanceof AnchorError) {
        expect(err.error.errorCode.code).to.equal("NotOperator");
      }
    }
    expect(threw).to.be.true;
  });

  // ── Test 7: Record job completion increments counter and reputation ────────
  it("records a job completion and increments jobs_completed", async () => {
    const [nodePda] = PublicKey.findProgramAddressSync(
      [Buffer.from("node"), operatorKp.publicKey.toBuffer()],
      program.programId
    );

    const nodeBefore = await program.account.nodeRecord.fetch(nodePda);
    const completedBefore = nodeBefore.jobsCompleted.toNumber();
    const reputationBefore = nodeBefore.reputation;

    // In production this is called via CPI from inference-market;
    // in tests the authorityKp plays the caller role
    await program.methods
      .recordCompletion()
      .accounts({
        node: nodePda,
        caller: authorityKp.publicKey,
      })
      .signers([authorityKp])
      .rpc();

    const nodeAfter = await program.account.nodeRecord.fetch(nodePda);
    expect(nodeAfter.jobsCompleted.toNumber()).to.equal(completedBefore + 1);
    // reputation should be +1, capped at 1000
    expect(nodeAfter.reputation).to.equal(Math.min(reputationBefore + 1, 1000));
  });

  // ── Test 8: Rejects registration with too many model IDs ──────────────────
  it("rejects registration with more than 8 model IDs", async () => {
    const extraKp = Keypair.generate();
    await fundKeypair(provider, extraKp, 5);

    const extraTokenAccount = await createAccount(
      provider.connection,
      extraKp,
      tokenMint,
      extraKp.publicKey
    );
    await mintTo(
      provider.connection,
      authorityKp,
      tokenMint,
      extraTokenAccount,
      authorityKp,
      MIN_STAKE.toNumber() * 2
    );

    const [nodePda] = PublicKey.findProgramAddressSync(
      [Buffer.from("node"), extraKp.publicKey.toBuffer()],
      program.programId
    );
    const stakeTokenAccount = await getAssociatedTokenAddress(
      tokenMint,
      nodePda,
      true
    );

    // 9 model IDs — one too many
    const tooManyModels = Array.from({ length: 9 }, () => randomModelId());

    let threw = false;
    try {
      await program.methods
        .registerNode("http://localhost:9090", 48, 2, tooManyModels, MIN_STAKE)
        .accounts({
          registry: registryPda,
          node: nodePda,
          stakeTokenAccount,
          operatorTokenAccount: extraTokenAccount,
          tokenMint,
          operator: extraKp.publicKey,
          tokenProgram: TOKEN_PROGRAM_ID,
          associatedTokenProgram: ASSOCIATED_TOKEN_PROGRAM_ID,
          systemProgram: SystemProgram.programId,
        })
        .signers([extraKp])
        .rpc();
    } catch (err: unknown) {
      threw = true;
      if (err instanceof AnchorError) {
        expect(err.error.errorCode.code).to.equal("TooManyModels");
      }
    }
    expect(threw).to.be.true;
  });
});
