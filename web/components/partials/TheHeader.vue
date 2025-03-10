<script setup lang="ts">
import { ref, onMounted, onBeforeUnmount } from 'vue';
import { Bars3Icon } from "@heroicons/vue/24/solid";
import { BsFillQuestionCircleFill } from 'vue-icons-plus/bs'
import { IoMail, IoLogoGithub } from 'vue-icons-plus/io';

const showMenu = ref(false);

function toggleNavbar() {
  showMenu.value = !showMenu.value;
}

function closeNavbar() {
  showMenu.value = false;
}

function handleOutsideClick(event: MouseEvent) {
  if (showMenu.value && !(event.target as HTMLElement | null)?.closest("nav")) {
    showMenu.value = false;
  }
}

const { $event } = useNuxtApp()
function startTour() {
  $event('startTour');
}

onMounted(() => {
  window.addEventListener("click", handleOutsideClick);
});

onBeforeUnmount(() => {
  window.removeEventListener("click", handleOutsideClick);
});
</script>

<template>
  <header
    class="py-4 px-2 md:px-4 lg:px-8 flex items-center justify-between font-semibold border-b-2 border-gray-200"
  >
    <nuxt-link
      to="/"
      class="flex justify-center items-center text-3xl font-bold"
    >
      <img id="logo" src="~/assets/img/koina-logo.svg" class="h-12" />
      <span class="ml-2"> Koina </span>
    </nuxt-link>

    <nav class="relative">
      <button
        class="cursor-pointer text-xl leading-none px-3 py-1 border border-solid border-transparent rounded bg-transparent block lg:hidden outline-none focus:outline-none"
        type="button"
        @click="toggleNavbar()"
      >
        <Bars3Icon class="h-6 w-6 text-black" />
      </button>

      <ul
        :class="[
          {
            hidden: !showMenu,
            flex: showMenu,
          },
          'lg:flex items-end flex-col lg:flex-row gap-12 px-6 lg:px-0 py-4 lg:py-0 absolute lg:static right-[-0.5rem] md:right-[-1rem] z-10 bg-white w-[100vw] lg:w-auto',
        ]"
        @click="closeNavbar()"
      >
        <li>
          <nuxt-link id="btn-docu" to="/docs"> Documentation </nuxt-link>
        </li>
        <li>
          <nuxt-link id="btn-comp-spec" to="/compare-spectra"> Compare Spectra </nuxt-link>
        </li>
        <li>
          <a id="contact-button" href="mailto:Ludwig.Lautenbacher@tum.de"> <IoMail/> </a>
        </li>
        <li>
          <nuxt-link to="https://github.com/wilhelm-lab/koina">
            <IoLogoGithub />
          </nuxt-link>
        </li>
        <li>
          <div @click="startTour" style="cursor:pointer"><BsFillQuestionCircleFill /></div>
        </li>
      </ul>
    </nav>
  </header>
</template>

<style scoped>
@tailwind base;
@tailwind components;
@tailwind utilities;

@layer base {
  a {
    @apply text-black;
  }
}
</style>
