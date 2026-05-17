import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  devIndicators: {
    appIsrStatus: false,
    buildActivity: false,
  },
  turbopack: {
    root: process.cwd(),
  },
};

export default nextConfig;
