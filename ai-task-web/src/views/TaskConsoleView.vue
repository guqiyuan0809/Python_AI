<script setup>
import { ref } from 'vue'
import { ElMessage } from 'element-plus'
import { submitAsyncTask } from '../api/task'
import { clearToken, getToken, setToken } from '../utils/auth'

const sessionId = ref('')
const message = ref('')
const submitting = ref(false)
const submittedTask = ref(null)
const accessToken = ref(getToken())

function saveAccessToken() {
  const token = accessToken.value.trim()
  if (!token) {
    ElMessage.warning('请先粘贴 Apifox 获取的临时 Token')
    return
  }
  setToken(token)
  ElMessage.success('临时 Token 已保存到当前浏览器')
}

function removeAccessToken() {
  clearToken()
  accessToken.value = ''
  ElMessage.success('临时 Token 已清除')
}

async function handleSubmit() {
  if (!getToken()) {
    ElMessage.warning('请先保存临时 Token')
    return
  }
  if (!sessionId.value.trim() || !message.value.trim()) {
    ElMessage.warning('请填写 Session ID 和问题')
    return
  }

  submitting.value = true
  submittedTask.value = null
  try {
    // 调用 Java 异步提交接口；此处只拿业务 taskId，不等待 AI 回答完成。
    submittedTask.value = await submitAsyncTask({
      sessionId: sessionId.value.trim(),
      message: message.value.trim(),
      historyLimit: 6,
    })
    ElMessage.success('异步任务已提交')
  } catch (error) {
    ElMessage.error(error.message)
  } finally {
    submitting.value = false
  }
}
</script>

<template>
  <section class="page-intro">
    <div>
      <p class="eyebrow">DAY 13 / ASYNC WORKFLOW</p>
      <h2>AI 异步会话任务</h2>
      <p>提交问题后，页面将通过 Java 查询 Python 任务状态。</p>
    </div>
    <span class="architecture-badge">Vue3 → Java → Python</span>
  </section>

  <el-alert
    title="本地开发鉴权"
    type="warning"
    :closable="false"
    show-icon
    description="此页面不实现登录。请粘贴 Apifox 获取的临时 Token；它只保存在当前浏览器 localStorage，不会写入源码。"
    class="token-tip"
  />

  <el-card class="task-card" shadow="never">
    <template #header>
      <div class="card-title">Java 临时 Token</div>
    </template>
    <el-input v-model="accessToken" type="password" show-password placeholder="粘贴临时 Token" />
    <div class="token-actions">
      <el-button type="primary" plain @click="saveAccessToken">保存 Token</el-button>
      <el-button @click="removeAccessToken">清除 Token</el-button>
    </div>
  </el-card>

  <el-card class="task-card" shadow="never">
    <template #header>
      <div class="card-title">提交任务</div>
    </template>

    <el-form label-position="top">
      <el-form-item label="Session ID">
        <el-input v-model="sessionId" placeholder="先填入已有会话 ID" />
      </el-form-item>
      <el-form-item label="问题">
        <el-input
          v-model="message"
          type="textarea"
          :rows="4"
          placeholder="例如：请总结当前会话"
        />
      </el-form-item>
      <el-button type="primary" :loading="submitting" @click="handleSubmit">
        提交异步任务
      </el-button>
    </el-form>
  </el-card>

  <el-card class="task-card" shadow="never">
    <template #header>
      <div class="card-title">任务状态预览</div>
    </template>
    <el-empty v-if="!submittedTask" description="提交任务后，这里显示 taskId 和任务状态" />
    <el-descriptions v-else :column="1" border>
      <el-descriptions-item label="Task ID">
        <code>{{ submittedTask.taskId }}</code>
      </el-descriptions-item>
      <el-descriptions-item label="初始状态">
        <el-tag type="warning">{{ submittedTask.status }}</el-tag>
      </el-descriptions-item>
      <el-descriptions-item label="说明">
        Python 已创建业务任务；下一阶段将轮询 Java 查询最终状态。
      </el-descriptions-item>
    </el-descriptions>
  </el-card>
</template>

<style scoped>
.page-intro {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 24px;
  margin-bottom: 24px;
}

.page-intro h2 {
  margin: 6px 0 8px;
  color: #111827;
  font-size: 30px;
}

.page-intro p:not(.eyebrow) {
  margin: 0;
  color: #64748b;
}

.architecture-badge {
  padding: 8px 12px;
  border: 1px solid #bfdbfe;
  border-radius: 999px;
  color: #1d4ed8;
  background: #eff6ff;
  font-size: 13px;
  white-space: nowrap;
}

.task-card {
  margin-bottom: 20px;
  border: none;
}

.token-tip {
  margin-bottom: 20px;
}

.token-actions {
  display: flex;
  gap: 12px;
  margin-top: 14px;
}

.card-title {
  color: #111827;
  font-weight: 700;
}

@media (max-width: 640px) {
  .app-header,
  .page-intro {
    flex-direction: column;
  }
}
</style>
