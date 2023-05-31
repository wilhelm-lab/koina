import type { Config } from 'tailwindcss'

export default <Partial<Config>>{
  theme: {
    extend: {
      colors: {
        primary: '#001bc8',
        secondary: '#f0ab00',
        'primary-dark': '#000C58'
      },
      backgroundImage: {
        'hero-pattern': "url('/images/hero-pattern.svg')",
      }
    }
  }
}
