<script setup>
import { RouterLink, RouterView } from 'vue-router'
</script>

<template>
  <div class="app-shell">
    <header class="app-header">
      <div>
        <p class="eyebrow">AI SERVICE CONSOLE</p>
        <h1>异步任务控制台</h1>
      </div>
      <nav class="app-nav" aria-label="主导航">
        <RouterLink to="/tasks">任务演示</RouterLink>
      </nav>
    </header>

    <main class="app-main">
      <RouterView />
    </main>
  </div>
</template>
