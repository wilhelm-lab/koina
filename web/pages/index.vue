<template>
  <div>
    <TheHero />

    <div class="py-16 px-2 max-w-5xl mx-auto">
      <h2 id="get-started" class="text-3xl mb-4">What is Koina?</h2>
      <p>
        Koina is a community driven endeavour to make AI models for life science
        research accessible. It is a service that tries to reduce the barrier of
        entry for using machine learning models in life science research by
        removing the need to set up and maintain the infrastructure required to
        run the models. The project is open-source and adheres to
        <a href="https://www.go-fair.org/fair-principles/">FAIR principles</a>.
      </p>
      <p>
        Koina is built on top of
        <a
          href="https://github.com/triton-inference-server/server/tree/v2.29.0"
        >
          Triton Inference Server</a
        >
        and enables hosting virtually any kind machine learning model. The
        models can be accessed via a
        <a href="https://en.wikipedia.org/wiki/Representational_state_transfer"
          >REST API</a
        >
        or <a href="https://grpc.io/">gRPC</a> and provides an
        <a href="https://swagger.io/specification/v3/">OpenAPI document</a> for
        documentation and making integration easy.
      </p>
      <p>
        Check out the <a href="/docs">documentation</a> to learn more about how
        to use Koina.
      </p>
    </div>

    <div class="bg-neutral-100">
      <div class="py-16 px-2 max-w-5xl mx-auto">
        <h2 id="get-involved" class="text-3xl mb-4">Get involved</h2>
        <p>
          There are two ways to get involved with Koina. You can either host an
          instance of Koina to make more open resources available to the
          community or you can add your own model to Koina. If you want to make
          your Koina instance available via
          <a href="https://koina.wilhelmlab.org">koina.wilhelmlab.org</a>,
          <a href="mailto:Ludwig.Lautenbacher@tum.de"
            >please contact us via E-Mail</a
          >.
        </p>
        <p class="mb-8">
          For more information on how to host an instance of Koina or add your
          own model, check out the links below.
        </p>
        <BaseButton to="/docs#overview--hosting-your-own-server">
          Host Koina
        </BaseButton>
        <BaseButton to="/docs#overview--adding-your-own-model" class="ml-4">
          Add your model
        </BaseButton>
      </div>
    </div>

    <div class="py-16 px-2 max-w-5xl mx-auto">
      <h2 class="text-3xl mb-4">Organizations hosting Koina</h2>
      <div class="flex justify-around flex-wrap">
        <a href="https://www1.ls.tum.de/en/compms/home/" class="mb-2">
          <img src="~/assets/img/tum-logo.png" alt="TUM logo" class="h-16" />
        </a>
        <a href="https://fgcz.ch/" class="ml-2 mb-2">
          <img
            src="~/assets/img/eth-uzh-logo.svg"
            alt="TUM logo"
            class="h-16"
          />
        </a>
        <a href="https://www.msaid.de/" class="ml-2 mb-2">
          <img src="~/assets/img/msaid-logo.webp" alt="TUM logo" class="h-16" />
        </a>
      </div>
    </div>
  </div>
</template>
