import { resolve } from "node:path";

/** @type {import('next').NextConfig} */
export default {
  output: "export",
  // The results table imports ../results/metrics.json from the Python side of the repo.
  turbopack: { root: resolve(import.meta.dirname, "..") },
};
