<script setup>
import { onBeforeUnmount, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { createSession, getTaskStatus, submitAsyncTask } from '../api/task'
import { clearToken, getToken, setToken } from '../utils/auth'

const sessionId = ref('')
const message = ref('')
const creatingSession = ref(false)
const submitting = ref(false)
const submittedTask = ref(null)
const polling = ref(false)
const accessToken = ref(getToken())
const terminalTaskStatuses = new Set(['success', 'error'])

let pollingTimer = null
let pollingTaskId = ''

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
  stopPolling()
  clearToken()
  accessToken.value = ''
  sessionId.value = ''
  submittedTask.value = null
  ElMessage.success('临时 Token 已清除')
}

function stopPolling() {
  if (pollingTimer) {
    window.clearTimeout(pollingTimer)
    pollingTimer = null
  }
  pollingTaskId = ''
  polling.value = false
}

function isTaskFinished(task) {
  return task?.finished === true || terminalTaskStatuses.has(task?.status)
}

async function createCurrentUserSession() {
  if (!getToken()) {
    ElMessage.warning('请先保存临时 Token')
    return ''
  }

  creatingSession.value = true
  try {
    const created = await createSession()
    sessionId.value = created.sessionId
    ElMessage.success('已为当前登录用户创建新会话')
    return sessionId.value
  } catch (error) {
    ElMessage.error(error.message)
    return ''
  } finally {
    creatingSession.value = false
  }
}

async function pollTaskStatus(taskId) {
  try {
    const task = await getTaskStatus(taskId)
    // 用户可能在请求未返回前提交了新任务或清除了 Token，旧响应不能覆盖新状态。
    if (pollingTaskId !== taskId) {
      return
    }
    submittedTask.value = task
    if (isTaskFinished(task)) {
      stopPolling()
      ElMessage.success(task.status === 'success' ? 'AI 任务已完成' : 'AI 任务执行失败')
      return
    }
    pollingTimer = window.setTimeout(() => pollTaskStatus(taskId), 1500)
  } catch (error) {
    if (pollingTaskId !== taskId) {
      return
    }
    submittedTask.value = {
      ...submittedTask.value,
      status: 'error',
      errorMessage: error.message,
    }
    stopPolling()
    ElMessage.error(`查询任务状态失败：${error.message}`)
  }
}

function startPolling(taskId) {
  stopPolling()
  pollingTaskId = taskId
  polling.value = true
  void pollTaskStatus(taskId)
}

async function handleSubmit() {
  if (!getToken()) {
    ElMessage.warning('请先保存临时 Token')
    return
  }
  if (!message.value.trim()) {
    ElMessage.warning('请填写问题')
    return
  }

  submitting.value = true
  stopPolling()
  submittedTask.value = null
  try {
    const activeSessionId = sessionId.value || await createCurrentUserSession()
    if (!activeSessionId) {
      return
    }
    // 调用 Java 异步提交接口；此处只拿业务 taskId，不等待 AI 回答完成。
    submittedTask.value = await submitAsyncTask({
      sessionId: activeSessionId,
      message: message.value.trim(),
      historyLimit: 6,
    })
    ElMessage.success('异步任务已提交')
    startPolling(submittedTask.value.taskId)
  } catch (error) {
    ElMessage.error(error.message)
  } finally {
    submitting.value = false
  }
}

function statusTagType(status) {
  if (status === 'success') return 'success'
  if (status === 'error') return 'danger'
  if (status === 'running') return 'primary'
  return 'warning'
}

onBeforeUnmount(stopPolling)
</script>

<template>
  <section class="page-intro">
    <div>
      <p class="eyebrow">DAY 13 / ASYNC WORKFLOW</p>
      <h2>AI 异步会话任务</h2>
        <p>页面自动创建当前用户会话，并通过 Java 轮询 Python 异步任务状态。</p>
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
        <div class="card-title">当前用户会话</div>
      </template>

      <el-input :model-value="sessionId" readonly placeholder="首次提交时会自动创建当前用户会话" />
      <div class="token-actions">
        <el-button :loading="creatingSession" @click="createCurrentUserSession">
          创建新会话
        </el-button>
      </div>
    </el-card>

    <el-card class="task-card" shadow="never">
      <template #header>
        <div class="card-title">提交异步任务</div>
      </template>

      <el-form label-position="top">
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
        <div class="card-title">任务状态与执行结果</div>
      </template>
      <el-empty v-if="!submittedTask" description="提交任务后，页面会自动轮询任务状态" />
      <el-descriptions v-else :column="1" border>
        <el-descriptions-item label="Task ID">
          <code>{{ submittedTask.taskId }}</code>
        </el-descriptions-item>
        <el-descriptions-item label="状态">
          <el-tag :type="statusTagType(submittedTask.status)">
            {{ submittedTask.status }}{{ polling ? '（轮询中）' : '' }}
          </el-tag>
        </el-descriptions-item>
        <el-descriptions-item v-if="submittedTask.traceId" label="Trace ID">
          <code>{{ submittedTask.traceId }}</code>
        </el-descriptions-item>
        <el-descriptions-item v-if="submittedTask.model" label="模型">
          {{ submittedTask.model }}
        </el-descriptions-item>
        <el-descriptions-item v-if="submittedTask.totalTokens !== null && submittedTask.totalTokens !== undefined" label="总 Token">
          {{ submittedTask.totalTokens }}
        </el-descriptions-item>
        <el-descriptions-item v-if="submittedTask.costMs !== null && submittedTask.costMs !== undefined" label="耗时">
          {{ submittedTask.costMs }} ms
        </el-descriptions-item>
        <el-descriptions-item v-if="submittedTask.resultText" label="AI 回复">
          <pre class="task-result">{{ submittedTask.resultText }}</pre>
        </el-descriptions-item>
        <el-descriptions-item v-if="submittedTask.errorMessage" label="错误信息">
          <span class="task-error">{{ submittedTask.errorMessage }}</span>
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

.task-result {
  margin: 0;
  white-space: pre-wrap;
  word-break: break-word;
  font-family: inherit;
  line-height: 1.7;
}

.task-error {
  color: #dc2626;
}

@media (max-width: 640px) {
  .app-header,
  .page-intro {
    flex-direction: column;
  }
}
</style>
