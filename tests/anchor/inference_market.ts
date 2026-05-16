import * as anchor from "@coral-xyz/anchor";
import { Program } from "@coral-xyz/anchor";
import { InferenceMarket } from "../target/types/inference_market";
import { assert } from "chai";
import {
  createMint,
  createAccount,
  mintTo,
  getAccount,
} from "@solana/spl-token";

describe("inference-market", () => {
  const provider = anchor.AnchorProvider.env();
  anchor.setProvider(provider);

  const program = anchor.workspace.InferenceMarket as Program<InferenceMarket>;

  // Keypairs
  const authority = provider.wallet as anchor.Wallet;
  const client = anchor.web3.Keypair.generate();
  const node = anchor.web3.Keypair.generate();

  let tokenMint: anchor.web3.PublicKey;
  let clientTokenAccount: anchor.web3.PublicKey;
  let nodeTokenAccount: anchor.web3.PublicKey;
  let treasury: anchor.web3.PublicKey;
  let treasuryTokenAccount: anchor.web3.PublicKey;

  const [marketPda] = anchor.web3.PublicKey.findProgramAddressSync(
    [Buffer.from("market")],
    program.programId
  );

  before(async () => {
    // Airdrop SOL to test accounts
    await provider.connection.requestAirdrop(client.publicKey, 2e9);
    await provider.connection.requestAirdrop(node.publicKey, 2e9);

    // Create token mint
    tokenMint = await createMint(
      provider.connection,
      (authority as any).payer,
      authority.publicKey,
      null,
      6
    );

    treasury = anchor.web3.Keypair.generate().publicKey;

    // Create token accounts
    clientTokenAccount = await createAccount(
      provider.connection,
      (authority as any).payer,
      tokenMint,
      client.publicKey
    );
    nodeTokenAccount = await createAccount(
      provider.connection,
      (authority as any).payer,
      tokenMint,
      node.publicKey
    );
    treasuryTokenAccount = await createAccount(
      provider.connection,
      (authority as any).payer,
      tokenMint,
      treasury
    );

    // Mint tokens to client
    await mintTo(
      provider.connection,
      (authority as any).payer,
      tokenMint,
      clientTokenAccount,
      authority.publicKey,
      1_000_000_000 // 1000 tokens
    );
  });

  it("Initializes the market", async () => {
    await program.methods
      .initialize(treasury)
      .accounts({
        market: marketPda,
        tokenMint,
        authority: authority.publicKey,
        systemProgram: anchor.web3.SystemProgram.programId,
      })
      .rpc();

    const market = await program.account.market.fetch(marketPda);
    assert.equal(market.treasury.toBase58(), treasury.toBase58());
    assert.equal(market.totalJobs.toNumber(), 0);
  });

  it("Posts a job with token escrow", async () => {
    const modelId = new Array(32).fill(1);
    const promptHash = new Array(32).fill(2);
    const promptCid = "bafybeiabc123";
    const maxTokens = 256;
    const paymentAmount = new anchor.BN(100_000); // 0.1 tokens
    const deadline = new anchor.BN(Math.floor(Date.now() / 1000) + 3600);

    const market = await program.account.market.fetch(marketPda);
    const jobId = market.totalJobs;

    const [jobPda] = anchor.web3.PublicKey.findProgramAddressSync(
      [Buffer.from("job"), jobId.toArrayLike(Buffer, "le", 8)],
      program.programId
    );

    // Escrow ATA is created by the instruction
    const escrowTokenAccount = await anchor.utils.token.associatedAddress({
      mint: tokenMint,
      owner: jobPda,
    });

    await program.methods
      .postJob(
        modelId,
        promptHash,
        promptCid,
        maxTokens,
        paymentAmount,
        deadline
      )
      .accounts({
        market: marketPda,
        job: jobPda,
        escrowTokenAccount,
        clientTokenAccount,
        tokenMint,
        client: client.publicKey,
      })
      .signers([client])
      .rpc();

    const job = await program.account.job.fetch(jobPda);
    assert.equal(job.paymentAmount.toNumber(), 100_000);
    assert.deepEqual(Object.keys(job.status), ["open"]);
  });

  it("Node claims the job", async () => {
    const market = await program.account.market.fetch(marketPda);
    const jobId = new anchor.BN(market.totalJobs.toNumber() - 1);

    const [jobPda] = anchor.web3.PublicKey.findProgramAddressSync(
      [Buffer.from("job"), jobId.toArrayLike(Buffer, "le", 8)],
      program.programId
    );

    await program.methods
      .claimJob()
      .accounts({
        job: jobPda,
        node: node.publicKey,
      })
      .signers([node])
      .rpc();

    const job = await program.account.job.fetch(jobPda);
    assert.deepEqual(Object.keys(job.status), ["inProgress"]);
    assert.equal(job.node.toBase58(), node.publicKey.toBase58());
  });
});
