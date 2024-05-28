import type { Config } from 'tailwindcss';
import Color from 'color';
import animate from "tailwindcss-animate";

const primary = Color('#001bc8')
const secondary = Color('#f0ab00')

const customColors = {
  primary: primary.string(),
  'primary-light': primary.lighten(0.2).string(),
  'primary-dark': primary.darken(0.2).string(),
  'primary-dark-2': primary.darken(0.4).string(),
  'primary-dark-3': primary.darken(0.6).string(),
  'primary-dark-4': primary.darken(0.8).string(),
  'primary-dark-5': primary.darken(1).string(),
  secondary: secondary.string(),
  'secondary-light': secondary.lighten(0.2).string(),
  'secondary-dark': secondary.darken(0.2).string(),
}

const safeColors = [
  ...Object.keys(customColors),
  'white',
  'black',
]

const safeColorClasses = [
  ...safeColors.map(color => `text-${color}`),
  ...safeColors.map(color => `bg-${color}`),
  ...safeColors.map(color => `border-${color}`),
  ...safeColors.map(color => `hover:text-${color}`),
  ...safeColors.map(color => `hover:bg-${color}`),
  ...safeColors.map(color => `hover:border-${color}`),
]

export default <Partial<Config>>{
  safelist: [
    ...safeColorClasses,
  ],
  theme: {
    container: {
      center: true,
      padding: "2rem",
      screens: {
        "2xl": "1400px",
      },
    },
    extend: {
      colors: customColors,
      backgroundImage: {
        'hero-pattern': "url('/images/hero-pattern.svg')",
      }
    },
    keyframes: {
      "accordion-down": {
        from: { height: 0 },
        to: { height: "var(--radix-accordion-content-height)" },
      },
      "accordion-up": {
        from: { height: "var(--radix-accordion-content-height)" },
        to: { height: 0 },
      },
    } as any,
    animation: {
      "accordion-down": "accordion-down 0.2s ease-out",
      "accordion-up": "accordion-up 0.2s ease-out",
    },
  },
  darkMode: ["class"],
  content: [
    './pages/**/*.{ts,tsx,vue}',
    './components/**/*.{ts,tsx,vue}',
    './app/**/*.{ts,tsx,vue}',
    './src/**/*.{ts,tsx,vue}',
  ],
  prefix: "",
  plugins: [animate],
}


/** @type {import('tailwindcss').Config} */
module.exports = {
}
