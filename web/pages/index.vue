<template>
  <div>
    <div
      class="py-16 flex items-center justify-center w-full bg-primary-dark bg-repeat bg-hero-pattern"
    >
      <div
        class="max-w-5xl flex flex-col items-center justify-center text-center"
      >
        <h1 class="text-9xl font-bold mb-4 text-white">Koina</h1>
        <p class="text-3xl text-white">
          A generic, open-source and accessible prediction service simplifying
          utilisation of machine learning models in life science research
        </p>
        <div class="flex justify-center items-center mt-8">
          <nuxt-link
            to="/docs"
            class="bg-secondary hover:bg-yellow-400 text-xl font-bold py-2 px-4 border-2 border-secondary rounded"
          >
            Get started
          </nuxt-link>
          <nuxt-link
            to="#get-involved"
            class="ml-8 hover:bg-secondary hover:text-white text-xl text-secondary font-bold py-2 px-4 border-2 border-secondary rounded"
          >
            Get involved
          </nuxt-link>
        </div>
      </div>
    </div>
    <div class="py-16 max-w-5xl mx-auto">
      <h2 id="get-started" class="text-3xl mb-4">Get started</h2>
      <p class="mb-4">
        Lorem ipsum, dolor sit amet consectetur adipisicing elit. Aut fugiat
        voluptates hic maxime vero, commodi laborum? Sed numquam quibusdam vel
        maiores nemo incidunt accusantium placeat. Impedit dolorum nemo rem
        tempora.
      </p>
      <p class="mb-8">
        Lorem ipsum dolor sit amet consectetur adipisicing elit. Eum quisquam
        praesentium commodi recusandae sed non vitae velit ab! Placeat,
        consequatur! Dolorem voluptates omnis vitae impedit amet voluptatem iste
        tenetur voluptate.
      </p>
      <nuxt-link
        to="/docs"
        class="bg-primary hover:bg-primary-400 text-white text-xl py-2 px-4 border-2 border-primary hover:border-blue-400 rounded mt-8"
      >
        Documentation
      </nuxt-link>
    </div>
    <div class="bg-neutral-100">
      <div class="py-16 max-w-5xl mx-auto">
        <h2 id="get-involved" class="text-3xl mb-4">Get involved</h2>
        <p class="mb-4">
          Lorem ipsum dolor sit amet consectetur adipisicing elit. Nostrum ea
          labore, reprehenderit enim cumque eos corrupti atque eaque repellat
          voluptatum sunt voluptas laudantium voluptatibus magni fugit
          asperiores facere esse perferendis.
        </p>
        <nuxt-link
          to="/docs"
          class="bg-primary hover:bg-primary-400 text-white text-xl py-2 px-4 border-2 border-primary hover:border-blue-400 rounded mt-8"
        >
          Host Koina
        </nuxt-link>
        <nuxt-link
          to="/docs"
          class="bg-primary hover:bg-primary-400 text-white text-xl py-2 px-4 border-2 border-primary hover:border-blue-400 rounded mt-8 ml-4"
        >
          Add your model
        </nuxt-link>
      </div>
    </div>
    <div class="py-16 max-w-5xl mx-auto">
      <h2 class="text-3xl mb-4">Organizations hosting Koina</h2>
      <div class="flex justify-between">
        <img src="~/assets/img/tum-logo.png" alt="TUM logo" class="h-16" />
        <img
          src="https://ethz.ch/etc/designs/ethz/img/header/ethz_logo_black.svg"
          alt="ETHZ logo"
          class="h-16"
        />
        <img
          src="https://www.msaid.de/hs-fs/hubfs/MSAID-Logo-Homepage-1.jpg?width=4080&height=1035&name=MSAID-Logo-Homepage-1.jpg"
          alt="MSAID logo"
          class="h-16"
        />
      </div>
    </div>
    <div class="bg-neutral-100">
      <div class="py-16 max-w-5xl mx-auto">
        <h2 class="text-3xl mb-4 block">What is Koina?</h2>
        <p>
          Lorem ipsum dolor, sit amet consectetur adipisicing elit. Nisi earum
          sunt, beatae quam culpa dignissimos. Ab, natus. Provident ab modi sunt
          cum nam vitae! Minus dolores dicta et omnis necessitatibus.
        </p>
      </div>
    </div>

    <footer class="py-8 px-2 md:px-4 lg:px-8 flex items-center justify-center">
      <div>
        <nuxt-link to="/privacy-policy">Privacy Policy</nuxt-link>
        <nuxt-link to="/imprint" class="ml-8">Imprint</nuxt-link>
        <nuxt-link to="/terms-of-use" class="ml-8">Terms of Use</nuxt-link>
      </div>
    </footer>
  </div>
</template>
