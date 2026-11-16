import type { Config } from 'tailwindcss';
import colors from 'tailwindcss/colors';

const config: Config = {
  darkMode: 'class',
  content: ['./app/**/*.{ts,tsx}', './components/**/*.{ts,tsx}', './lib/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // Warm orange accent on neutral greys and near-black.
        brand: {
          50: '#fff4ee',
          100: '#ffe6d9',
          200: '#ffc9ae',
          300: '#ffa47a',
          400: '#fb7d48',
          500: '#f26430',
          600: '#e04f1d',
          700: '#b93d15',
          800: '#933317',
          900: '#772d16',
          950: '#40140a',
        },
        // Every slate-* class renders as a neutral (non-blue) grey.
        slate: colors.neutral,
      },
      fontFamily: {
        sans: [
          'ui-sans-serif',
          'system-ui',
          '-apple-system',
          'Segoe UI',
          'Roboto',
          'Helvetica Neue',
          'Arial',
          'sans-serif',
        ],
      },
      boxShadow: {
        card: '0 1px 2px 0 rgb(0 0 0 / 0.04)',
      },
    },
  },
  plugins: [],
};

export default config;
