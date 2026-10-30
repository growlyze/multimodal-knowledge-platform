/** @type {import('next').NextConfig} */
const nextConfig = {
  // The whole dashboard is deployed behind Caddy under the /kb path prefix
  // (Caddyfile: handle /kb/* -> dashboard:3000). Route handlers under
  // app/api/* are therefore reached at /kb/api/*.
  basePath: '/kb',
  // Dockerfile copies .next/standalone (server.js) + .next/static + public.
  output: 'standalone',
  reactStrictMode: true,
  poweredByHeader: false,
  async headers() {
    return [
      {
        source: '/:path*',
        headers: [
          { key: 'X-Frame-Options', value: 'DENY' },
          { key: 'X-Content-Type-Options', value: 'nosniff' },
          { key: 'Referrer-Policy', value: 'same-origin' },
          { key: 'Permissions-Policy', value: 'camera=(), microphone=(), geolocation=()' },
        ],
      },
    ];
  },
};

module.exports = nextConfig;
