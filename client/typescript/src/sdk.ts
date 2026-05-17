import { AnchorProvider, Program } from "@coral-xyz/anchor";
import { Connection, PublicKey } from "@solana/web3.js";
import type { AnchorWallet } from "@solana/wallet-adapter-react";

export const INFERENCE_MARKET_PROGRAM_ID = new PublicKey(
  "5YQyZqXkJHy6V3JMxKqXyLqfP9V2A3j8Rk7mN4oD1eW"
);
export const COMPUTE_REGISTRY_PROGRAM_ID = new PublicKey(
  "8KpR2mT6uLqVwNzS4eBfY9oA3cJ7iGxH1nD5sW0qF2M"
);
export const GOVERNANCE_PROGRAM_ID = new PublicKey(
  "3CvE7tX9rMwPfBgY2nKjH6oL4sQ8uZaD5mR1iW0eN9T"
);

export const MODEL_IDS: Record<string, Uint8Array> = {
  "llama-3.2-1b": sha256("meta-llama/Llama-3.2-1B"),
  "llama-3.2-3b": sha256("meta-llama/Llama-3.2-3B"),
  "llama-3.1-8b": sha256("meta-llama/Llama-3.1-8B"),
  "mistral-7b": sha256("mistralai/Mistral-7B-v0.3"),
};

export function createProvider(
  connection: Connection,
  wallet: AnchorWallet
): AnchorProvider {
  return new AnchorProvider(connection, wallet, { commitment: "confirmed" });
}

function sha256(input: string): Uint8Array {
  // Node.js / browser-compatible SHA-256 (simplified — use crypto.subtle in production)
  const encoder = new TextEncoder();
  const data = encoder.encode(input);
  // Placeholder: real implementation uses SubtleCrypto
  return new Uint8Array(32).fill(0); // TODO: replace with real sha256
}
