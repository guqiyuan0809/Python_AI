# Python 后端学习路线

## 这条路线是什么？

后端开发就像餐厅的厨房——你在 App 上点了一份外卖，是后端帮你查库存、算价格、生成订单、通知骑手。用户看不到后端，但离了后端什么都转不了。

Python 是后端入门门槛最低的语言，语法简洁、生态丰富，三大主流框架（Flask、FastAPI、Django）覆盖了从学习到生产的所有场景。更重要的是，Python 还能无缝衔接 AI、数据分析、爬虫方向，技能可迁移性极强。

这条路线按照 **基础 → 前置知识 → 中间件 → 框架 → 项目实战 → 工程化** 的顺序编排，每一步都是下一步的地基。

### 适合谁？

- 零基础想学后端开发的同学
- 学过 Python 基础语法，想往后端方向深入的同学

---

## 学习路线

### 🟢 第一阶段：Python 基础（预计 3-4 周）

**目标：掌握 Python 核心语法和编程思维，能独立写出结构清晰的小程序。**

> 💡 这一阶段是地基中的地基。很多人急着学框架，结果连装饰器和生成器都说不清楚，面试直接挂。把基础打牢，后面学什么都快。

1. **基础语法** ⭐⭐⭐ 🎯
   - 变量、数据类型（str、int、float、bool）
   - 容器类型：list、dict、tuple、set 及其常用操作
   - 条件判断（if/elif/else）、循环（for、while）、推导式
   - 函数定义、参数（默认参数、*args、**kwargs）
   - 字符串格式化：f-string、format
   - 「一句话总结」：Python 语法就像写英语，`if age > 18: print("成年了")` 一看就懂
   - ⚠️ 踩坑：Python 用缩进表示代码块，Tab 和空格混用会报错，而且报错信息经常不指向真正出问题的那一行

2. **面向对象编程（OOP）** ⭐⭐⭐ 🎯
   - 类和对象、`__init__` 构造方法
   - 继承、多态、封装
   - 魔术方法：`__str__`、`__repr__`、`__len__`、`__eq__`
   - 类方法、静态方法、属性装饰器 `@property`
   - 「一句话总结」：OOP 就是把数据和操作数据的方法打包成一个"对象"
   - ⚠️ 踩坑：Python 没有真正的私有属性，`__name` 只是名称改写（name mangling），别把它当成 Java 的 private

3. **高级特性** ⭐⭐⭐ 🎯
   - **装饰器（decorator）**：理解闭包、`@` 语法糖、带参数的装饰器 —— 面试必考
   - **迭代器和生成器**：`yield`、惰性求值、节省内存
   - **上下文管理器**：`with` 语句、`__enter__` / `__exit__`
   - **类型提示（Type Hints）**：`typing` 模块、函数参数和返回值类型标注
   - 「一句话总结」：这些特性是 Python 的精华，也是区分"会写 Python"和"写好 Python"的分水岭

4. **异常处理和文件操作** ⭐⭐
   - `try / except / else / finally`
   - 自定义异常类
   - 文件读写：`open`、`with` 语句、`pathlib`
   - JSON 读写：`json.loads` / `json.dumps`

5. **常用标准库** ⭐⭐
   - `os`、`sys`：系统交互
   - `datetime`：日期时间处理
   - `re`：正则表达式
   - `collections`：Counter、defaultdict、OrderedDict
   - `functools`：partial、lru_cache、wraps

6. **虚拟环境和包管理** ⭐⭐⭐
   - `venv` 创建虚拟环境
   - `pip` 安装和管理依赖
   - `requirements.txt` 和 `pip freeze`
   - 了解 Poetry / PDM 等现代包管理工具
   - 「一句话总结」：虚拟环境就是给每个项目一个独立的"沙盒"，互不干扰
   - ⚠️ 踩坑：不用虚拟环境，所有包全局安装，项目 A 要 requests 2.28，项目 B 要 requests 2.31，直接炸

**练手建议**：
- 写一个命令行版的待办事项管理工具（增删改查 + 文件持久化）
- 写一个文件批量重命名/整理脚本
- 用装饰器实现一个函数计时器和日志记录器

---

### 🔵 第二阶段：后端前置知识（预计 2-3 周）

**目标：理解 Web 开发的底层原理，为学习框架做好知识储备。**

> 💡 很多人跳过这一步直接学框架，结果知其然不知其所以然。比如不懂 HTTP 协议，遇到跨域问题就两眼一黑；不懂 TCP，面试一问三不知。花两三周搞清楚这些概念，后面学框架会事半功倍。

7. **计算机网络基础** ⭐⭐⭐ 🎯
   - **TCP/IP 基础**：三次握手、四次挥手、TCP 和 UDP 的区别
   - **DNS 解析**：域名是怎么变成 IP 地址的
   - **端口**：为什么后端服务要监听端口，常见端口号（80、443、3306、6379、8000）
   - 「一句话总结」：网络协议就是计算机之间的"通信规则"，TCP 保证数据不丢，UDP 追求速度
   - ⚠️ 面试高频：TCP 三次握手/四次挥手的过程和原因，几乎每场面试都会问

8. **HTTP 协议** ⭐⭐⭐ 🎯
   - **请求方法**：GET、POST、PUT、PATCH、DELETE 的语义和区别
   - **状态码**：200、201、204、301、302、400、401、403、404、500、502、503
   - **请求和响应结构**：请求行、请求头、请求体、响应头、响应体
   - **Content-Type**：application/json、multipart/form-data、application/x-www-form-urlencoded
   - **Cookie 和 Session**：无状态协议如何保持用户登录状态
   - **HTTPS**：SSL/TLS 加密的基本原理、为什么比 HTTP 安全
   - **CORS（跨域资源共享）**：什么是跨域、为什么会跨域、怎么解决
   - 「一句话总结」：HTTP 就是浏览器和服务器之间的"对话格式"——请求怎么写、响应怎么回、状态码代表什么意思

9. **RESTful API 设计规范** ⭐⭐⭐
   - URL 设计：名词复数、层级关系（`/users/123/orders`）
   - HTTP 方法对应 CRUD：GET=查、POST=增、PUT=全量改、PATCH=部分改、DELETE=删
   - 统一响应格式：`{ "code": 200, "message": "success", "data": {...} }`
   - 分页、排序、过滤、搜索的设计
   - API 版本管理：`/api/v1/users`
   - 「一句话总结」：RESTful 就是一套"怎么设计好用的 API"的约定

10. **认证与授权基础** ⭐⭐⭐ 🎯
    - **Session vs Token**：有状态 vs 无状态，各自的优缺点
    - **JWT（JSON Web Token）**：结构（Header.Payload.Signature）、生成和验证流程
    - **OAuth 2.0**：第三方登录的基本流程（授权码模式）
    - **密码安全**：为什么不能明文存密码、bcrypt / argon2 哈希
    - 「一句话总结」：认证 = "你是谁"，授权 = "你能干什么"
    - ⚠️ 踩坑：千万不要自己"发明"加密算法，用成熟的库（bcrypt、passlib）就对了

**练手建议**：
- 用 `curl` 或 Postman 手动发送 HTTP 请求，观察请求和响应的细节
- 尝试用 Python 的 `http.server` 模块写一个最简单的 HTTP 服务器
- 画一张 HTTP 请求的完整流程图（DNS → TCP 连接 → 发送请求 → 接收响应）

---

### 🟡 第三阶段：中间件技术（预计 3-4 周）

**目标：掌握后端开发必备的数据库和缓存技术，为框架开发提供数据支撑。**

> 💡 后端 80% 的工作都在和数据打交道。框架只是"壳"，数据库才是"芯"。MySQL 和 Redis 是后端面试的重灾区，必须扎实掌握。

11. **MySQL** ⭐⭐⭐ 🎯
    - **SQL 基础**：CREATE TABLE、INSERT、SELECT、UPDATE、DELETE
    - **查询进阶**：WHERE、ORDER BY、GROUP BY、HAVING、LIMIT
    - **多表查询**：INNER JOIN、LEFT JOIN、RIGHT JOIN、子查询
    - **索引**：什么是索引、为什么能加速查询、B+ 树的基本概念、联合索引和最左前缀原则
    - **事务**：ACID 特性、事务隔离级别（读未提交、读已提交、可重复读、串行化）
    - **表设计**：三大范式、一对一/一对多/多对多关系设计、外键
    - **慢查询优化**：EXPLAIN 分析执行计划、避免全表扫描
    - 「一句话总结」：MySQL 是存数据的"仓库"，SQL 是操作仓库的"语言"
    - ⚠️ 面试高频：索引原理、事务隔离级别、慢查询优化 —— 这三个是 MySQL 面试的"三座大山"

12. **Redis** ⭐⭐⭐ 🎯
    - **五种基本数据结构**：String、Hash、List、Set、Sorted Set（ZSet）
    - **常见使用场景**：缓存热点数据、Session 存储、排行榜（ZSet）、分布式锁（SETNX）、消息队列（List/Stream）
    - **缓存策略**：缓存穿透、缓存击穿、缓存雪崩的概念和解决方案
    - **过期策略和淘汰策略**：TTL、惰性删除、定期删除、LRU/LFU
    - **持久化**：RDB 快照 vs AOF 日志
    - 「一句话总结」：Redis 是"内存版的超级字典"，读写速度是 MySQL 的 100 倍
    - ⚠️ 踩坑：Redis 是单线程模型，别在里面跑耗时操作（比如 KEYS * 在生产环境会直接卡死）

13. **Python ORM 框架** ⭐⭐⭐
    - **SQLAlchemy**：模型定义、增删改查、关联关系、Session 管理
    - **Tortoise ORM**：异步 ORM，配合 FastAPI 使用
    - **Alembic**：数据库迁移工具（类似 Django 的 migrate）
    - 「一句话总结」：ORM 让你用 Python 代码操作数据库，不用手写 SQL
    - ⚠️ 踩坑：ORM 很方便，但生成的 SQL 可能很烂。一定要学会看 ORM 生成的 SQL（打开日志），复杂查询该手写 SQL 就手写

14. **消息队列（了解）** ⭐⭐
    - 消息队列的作用：异步解耦、削峰填谷、可靠投递
    - **Celery**：Python 最流行的异步任务队列（发邮件、生成报表、定时任务）
    - **RabbitMQ / Redis 作为消息代理**：基本概念和使用场景
    - 「一句话总结」：消息队列就是"把耗时的活先排个队，后台慢慢处理，不让用户干等"

**练手建议**：
- 在 MySQL 中设计一个简单的电商数据库（用户表、商品表、订单表、订单详情表）
- 用 Redis 实现一个排行榜：用 ZSet 存分数，ZREVRANGE 取 Top 10
- 用 SQLAlchemy 连接 MySQL，写一套完整的 CRUD 操作

---

### 🟠 第四阶段：三大主流框架（预计 4-6 周）

**目标：掌握三大框架的核心用法，能用任意一个框架独立开发完整的 RESTful API。**

> 💡 三个框架各有定位：Flask 帮你理解原理、FastAPI 是现代项目首选、Django 适合快速交付。建议按 Flask → FastAPI → Django 的顺序学，先轻后重。

15. **Flask（入门框架）** ⭐⭐⭐（1-2 周）
    - **核心概念**：路由、视图函数、请求对象、响应对象
    - **模板引擎**：Jinja2 基础（虽然做 API 用得少，但要了解）
    - **蓝图（Blueprint）**：项目模块化拆分
    - **中间件和钩子**：`before_request`、`after_request`、`errorhandler`
    - **Flask 扩展**：Flask-SQLAlchemy、Flask-Migrate、Flask-JWT-Extended、Flask-CORS
    - **项目结构**：从单文件到工厂函数模式（`create_app`）
    - 「一句话总结」：Flask 是一个"空房子"，你需要什么自己装，灵活但要自己动手
    - 💡 学习建议：Flask 最适合理解 Web 框架的底层原理，建议第一个框架学 Flask

16. **FastAPI（现代首选）** ⭐⭐⭐ 🎯（2-3 周）
    - **核心概念**：路由、路径参数、查询参数、请求体
    - **Pydantic 数据验证**：BaseModel、Field、validator —— 自动校验请求数据
    - **依赖注入**：`Depends()`，优雅地管理数据库连接、认证、权限
    - **异步支持**：`async / await`、异步数据库操作
    - **自动 API 文档**：Swagger UI（`/docs`）和 ReDoc（`/redoc`），写完代码文档就有了
    - **中间件**：CORS 中间件、自定义中间件、异常处理
    - **WebSocket**：实时通信支持
    - **项目结构**：路由分层、Service 层、Repository 层
    - 「一句话总结」：FastAPI 是 Python 后端的"现代标配"——快、类型安全、自带文档
    - ⚠️ 踩坑：FastAPI 是异步框架，用同步的数据库驱动（如 pymysql）会阻塞事件循环。要用异步驱动（asyncpg、aiomysql）或者用 `run_in_executor`

17. **Django（全功能框架）** ⭐⭐⭐（1-2 周）
    - **MTV 架构**：Model（数据模型）→ Template（模板）→ View（视图）
    - **Django ORM**：模型定义、QuerySet API、数据库迁移（makemigrations / migrate）
    - **Admin 后台**：自动生成管理后台，增删改查不用写一行前端代码
    - **认证系统**：自带用户注册、登录、权限管理
    - **Django REST Framework（DRF）**：序列化器、视图集、路由器、分页、过滤
    - **信号（Signals）**：模型保存前/后触发自定义逻辑
    - 「一句话总结」：Django 是"拎包入住的精装房"，ORM、Admin、认证、权限全都自带
    - 💡 学习建议：不需要像 Flask/FastAPI 那样深入，了解架构思想和 DRF 的用法即可

**三大框架对比**：

| 特性 | Flask | FastAPI | Django |
|------|-------|---------|--------|
| 定位 | 微框架 | 现代高性能框架 | 全功能框架 |
| 学习曲线 | 低 | 中 | 中高 |
| 异步支持 | 需扩展 | 原生支持 | 3.1+ 支持 |
| API 文档 | 需扩展 | 自动生成 | DRF 支持 |
| ORM | 需扩展 | 需搭配 | 自带 |
| Admin 后台 | 无 | 无 | 自带 |
| 适合场景 | 小项目、学习 | API 服务、现代项目 | 快速交付、全栈项目 |
| 社区生态 | 成熟 | 快速增长 | 最成熟 |

**练手建议**：
- Flask：做一个简单的博客 API（文章的增删改查 + 用户认证）
- FastAPI：做一个完整的 RESTful API 项目（用户注册登录 + JWT 认证 + CRUD + 分页 + 数据验证）
- Django + DRF：做一个图书管理系统后台（体验 Admin + DRF 的开发效率）

---

### 🔴 第五阶段：项目实战（预计 4-6 周）

**目标：独立完成一个完整的后端项目，从需求分析到部署上线，积累真实的项目经验。**

> 💡 这是整条路线最重要的阶段。简历上写一堆"精通 xxx"没用，面试官要看的是你做过什么项目、遇到过什么问题、怎么解决的。一个质量高的项目，胜过十个"增删改查"。

#### 项目建议

**推荐项目一：电商网站**（经典全栈项目）

```
功能清单：
├── 用户模块：注册、登录、JWT 认证、个人信息、收货地址管理
├── 商品模块：商品列表（分页+搜索+筛选）、商品详情、分类管理
├── 购物车：添加、修改数量、删除、选中结算
├── 订单模块：创建订单、订单状态流转（待支付→已支付→已发货→已完成）、取消订单
├── 支付模块：模拟支付接口 / 对接支付宝沙箱
├── 缓存优化：商品详情页 Redis 缓存、库存扣减防超卖（Redis + 分布式锁）
├── 异步任务：Celery 处理订单超时自动取消、发送订单通知邮件
└── 后台管理：商品上下架、订单管理、数据统计看板
```

**推荐项目二：AI 简历分析系统**（体现 AI 应用能力）

```
功能清单：
├── 用户模块：注册、登录、JWT 认证、历史分析记录
├── 简历上传：支持 PDF / Word 格式上传、文件解析提取文本
├── AI 分析：调用大模型 API（OpenAI / 通义千问 / DeepSeek）对简历进行多维度分析
│   ├── 内容完整性评估（教育、经历、技能是否齐全）
│   ├── 亮点与不足分析
│   ├── 岗位匹配度打分（用户输入目标岗位 JD）
│   └── 优化建议生成
├── Prompt 工程：设计结构化 Prompt，保证输出格式可控、质量稳定
├── 异步处理：Celery 异步调用 AI 接口，避免请求超时
├── 结果存储：分析结果持久化、支持历史对比
└── API 文档：Swagger UI 自动生成
```

**推荐项目三：任务调度监控平台**（偏工程化）

```
功能清单：
├── 用户模块：注册、登录、角色权限（管理员 / 普通用户）
├── 任务管理：创建定时任务 / 一次性任务、编辑、暂停、删除
├── 任务调度：基于 Celery Beat 的 Cron 定时调度、支持手动触发执行
├── 执行记录：每次执行的状态（成功/失败）、耗时、输出日志、错误信息
├── 监控看板：任务成功率、失败率、平均耗时、近期执行趋势图
├── 告警通知：任务连续失败时发送邮件 / Webhook 告警
├── 日志系统：结构化日志、按任务 ID 检索执行日志
└── Docker 部署：docker-compose 一键启动（FastAPI + Celery + Redis + MySQL）
```

#### 项目要有"亮点"

面试官看过太多纯 CRUD 项目了，你的项目需要至少包含 2-3 个技术亮点：

- **缓存优化**：用 Redis 缓存热门数据，对比缓存前后的响应时间
- **异步任务**：用 Celery 处理邮件发送、报表生成等耗时操作
- **JWT 认证**：完整的 Token 刷新机制（Access Token + Refresh Token）
- **分布式锁**：用 Redis 实现防止重复提交
- **日志系统**：结构化日志、请求链路追踪
- **单元测试**：pytest 测试覆盖核心逻辑，覆盖率 > 70%
- **API 文档**：自动生成的 Swagger 文档
- **Docker 部署**：Dockerfile + docker-compose 一键启动

---

### 🟣 第六阶段：工程化能力（预计 3-4 周）

**目标：掌握后端开发的工程化工具链，具备独立部署和运维项目的能力。**

> 💡 能写代码只是第一步，能把代码部署上线、稳定运行、持续迭代，才是一个合格的后端工程师。这些工程化技能在工作中天天用，面试也经常问。

> 工程化能力是指把代码从"能跑"变成"能上线、能维护、能协作"的一整套技能。


18. **Linux** ⭐⭐⭐ 🎯
    - **基础命令**：ls、cd、mkdir、cp、mv、rm、cat、grep、find、chmod、chown
    - **文件和权限**：rwx 权限模型、用户和用户组
    - **进程管理**：ps、top、kill、nohup、systemctl
    - **网络相关**：curl、wget、netstat、ss、ping、telnet
    - **文本处理**：grep、awk、sed、管道 `|` 和重定向 `>`
    - **Shell 脚本基础**：变量、条件、循环、定时任务（crontab）
    - 「一句话总结」：后端服务 90% 跑在 Linux 上，不会 Linux 的后端等于半残
    - ⚠️ 踩坑：`rm -rf /` 能删掉整个系统，永远不要在 root 下随便执行 rm 命令

19. **Git 版本控制** ⭐⭐⭐ 🎯
    - **基础操作**：init、clone、add、commit、push、pull、status、log、diff
    - **分支管理**：branch、checkout、merge、rebase
    - **团队协作**：Fork、Pull Request、Code Review 流程
    - **Git Flow**：main / develop / feature / hotfix 分支模型
    - **.gitignore**：忽略不需要提交的文件（虚拟环境、.env、__pycache__）
    - **冲突解决**：merge conflict 的处理方法
    - 「一句话总结」：Git 是代码的"时光机"，每一次 commit 都是一个存档点

20. **Docker** ⭐⭐⭐ 🎯
    - **核心概念**：镜像（Image）、容器（Container）、Dockerfile、仓库（Registry）
    - **Dockerfile 编写**：FROM、COPY、RUN、CMD、EXPOSE、多阶段构建
    - **docker-compose**：多容器编排（Web 服务 + MySQL + Redis 一键启动）
    - **数据持久化**：Volume 挂载
    - **网络**：容器间通信、端口映射
    - **实践**：把你的项目 Docker 化，写 docker-compose.yml 一键启动整个环境
    - 「一句话总结」：Docker 把你的应用和环境打包成一个"集装箱"，到哪都能跑

21. **Nginx** ⭐⭐⭐
    - **反向代理**：把请求转发给后端服务（Gunicorn/Uvicorn）
    - **静态文件服务**：前端文件托管
    - **负载均衡**：多个后端实例之间分发请求
    - **HTTPS 配置**：SSL 证书配置、HTTP 自动跳转 HTTPS
    - **常用配置**：server、location、proxy_pass、gzip
    - 「一句话总结」：Nginx 是后端服务的"门卫"，所有请求先经过它

22. **CI/CD 持续集成与部署** ⭐⭐
    - **概念**：代码推上去 → 自动跑测试 → 自动构建 → 自动部署
    - **GitHub Actions**：编写 workflow 文件、自动化测试和部署
    - **基本流程**：push → lint 检查 → 单元测试 → 构建 Docker 镜像 → 部署到服务器
    - 「一句话总结」：CI/CD 就是"让机器代替你做那些重复的部署操作"
    - 💡 建议：先用 GitHub Actions（免费、简单），有需要再学 Jenkins / GitLab CI

23. **生产部署实践** ⭐⭐⭐
    - **WSGI/ASGI 服务器**：Gunicorn（WSGI）、Uvicorn（ASGI）
    - **进程管理**：Supervisor、systemd
    - **日志管理**：日志轮转、结构化日志、ELK（了解）
    - **环境管理**：开发环境 / 测试环境 / 生产环境的配置分离、.env 文件
    - **安全**：不要把密钥提交到 Git、CORS 配置、SQL 注入防护、XSS 防护
    - ⚠️ 踩坑：开发环境用 `uvicorn --reload` 没问题，生产环境一定要用 Gunicorn + Uvicorn Worker，否则性能和稳定性都没保障

24. **微服务（了解即可）** ⭐⭐
    - **核心思想**：把一个大应用拆成多个独立的小服务，每个服务负责一个业务模块，独立开发、独立部署、松耦合
    - **服务间通信**：HTTP REST 调用、gRPC（高性能 RPC 框架）
    - **容器编排**：Docker Compose（开发环境）、Kubernetes / K8s（生产环境，了解概念即可）
    - **服务治理（了解概念）**：服务注册与发现、负载均衡、熔断降级、链路追踪
    - 「一句话总结」：微服务是架构思想，不是某个框架。学完这条路线你已经具备拆服务的能力（FastAPI + Docker + Nginx），面试能讲清楚"什么时候该拆、怎么拆、服务间怎么通信"就够了
---

## 完整学习时间线

| 阶段 | 内容 | 预计时间 |
|------|------|----------|
| 🟢 第一阶段 | Python 基础 | 3-4 周 |
| 🔵 第二阶段 | 后端前置知识（网络、HTTP、认证） | 2-3 周 |
| 🟡 第三阶段 | 中间件技术（MySQL、Redis、ORM） | 3-4 周 |
| 🟠 第四阶段 | 三大框架（Flask → FastAPI → Django） | 4-6 周 |
| 🔴 第五阶段 | 项目实战 | 4-6 周 |
| 🟣 第六阶段 | 工程化（Linux、Git、Docker、Nginx、CI/CD） | 3-4 周 |
| **总计** | | **约 5-6 个月** |

> 💡 工程化能力（Linux、Git、Docker）不一定要放到最后学。Git 在第一阶段就应该开始用，Linux 基础命令也可以边学边用。这里把它们集中放在第六阶段，是为了系统地梳理和深入。

> 💡然后就是虽然上面标记时间为5-6个月，但是如果你掌握了我之前讲的四个高效学习方法，尤其是AI辅助法，我认为，3个月的时间学完这些，完全没有问题，而且并不要求你学的非常好，这些该学的你都学了，把基础打牢，多做点项目，就达到实习的程度，公司也不会要求你实习生的后端水平很好，差不多就行，而且，你进入公司后，开发模式大概率是Vibe Coding开发模式，基本上也不用太多代码，但是你得能看懂代码，能够审查出代码有没有问题，这样就可以了。从现在开始，3，4，5 三个月，刚刚好，然后六月找实习，找到实习后，就可以在暑假进行实习了。

## 学习资源推荐

| 资源名称 | 资源链接/描述 |
| :--- | :--- |
| **Python 基础** | [Python 语言程序设计](https://www.yuwuzheng.com/resources/python%E8%AF%AD%E8%A8%80%E7%A8%8B%E5%BA%8F%E8%AE%BE%E8%AE%A1-3073b4) |
| **Python 基础视频教程** | [B站视频，强烈推荐，B站播放量第一](https://www.bilibili.com/video/BV1wD4y1o7AS/?spm_id_from=333.337.search-card.all.click) |
| **Flask** | [B站视频](https://www.bilibili.com/video/BV11PoTYkEE1/?spm_id_from=333.337.search-card.all.click) |
| **FastAPI** | [推荐教程和项目](https://www.bilibili.com/video/BV1zV2QBtE39/?spm_id_from=333.337.search-card.all.click&vd_source=4ba6eb1cd3e27af7e72a5dad12084f96) |
| **Django** | [B站视频](https://www.bilibili.com/video/BV1rT4y1v7uQ/?spm_id_from=333.337.search-card.all.click) |
| **MySQL** | 推荐使用本站资源，作者认为，没有必要花大量时间去看视频学习Mysql，视频讲的也很基础，不如通过本站资源快速掌握基础，应对实习绰绰有余<br>[本站资源教程](https://www.yuwuzheng.com/resources/mysql%E6%95%99%E7%A8%8B) |
| **Redis** | 和Mysql一样，也是推荐使用本站资源快速掌握Redis<br>[本站资源教程](https://www.yuwuzheng.com/resources/redis%E6%95%99%E7%A8%8B) |
| **Linux** | Linux系统会基本操作就行，推荐本站资源，通过本站资源可以快速掌握<br>[本站资源教程](https://www.yuwuzheng.com/resources/linux%E6%95%99%E7%A8%8B) |
| **Docker** | Docker也没必要看视频学习，太费时间，通过本站资源教程，快速学习，应对实习是没有问题的<br>[本站资源教程](https://www.yuwuzheng.com/resources/docker%E6%95%99%E7%A8%8B) |
| **Git** | 通过本站资源，快速学习Git，足以应对实习中的日常开发<br>[本站资源教程](https://www.yuwuzheng.com/resources/git%E6%95%99%E7%A8%8B) |
| **B站推荐项目实战 - Django** | [Django天天生鲜项目](https://www.bilibili.com/video/BV1vt41147K8/?spm_id_from=333.1387.favlist.content.click&vd_source=4ba6eb1cd3e27af7e72a5dad12084f96) |
| **B站推荐项目实战 - Flask** | [Flask项目推荐](https://www.bilibili.com/video/BV1uZ4y1p77F/?spm_id_from=333.337.search-card.all.click&vd_source=4ba6eb1cd3e27af7e72a5dad12084f96) |

---

## 作者寄语

Python 后端入门真的不难。Python 的语法简洁优雅，FastAPI 的开发体验在所有后端框架中数一数二

很多人纠结"Python 后端岗位不如 Java 多"——这是事实。但换个角度看：Python 后端的竞争也没有 Java 那么卷，而且 Python + AI 的复合型人才现在非常抢手。

我的建议是：**先用 Python 入门后端，把 HTTP、数据库、认证、部署这些核心概念吃透。** 这些知识是通用的，以后想转 Java、Go 也能快速上手。

最后记住一句话：**后端的核心竞争力不是会多少框架，而是解决问题的能力。** 框架会过时，但你理解的原理、积累的经验、培养的工程化思维，永远不会过时。

举个例子，像AI大模型有很多，你会用deepseek去解决工作中的问题，换成chatgpt，难道就不会了吗？当然是会用的，而且很快会能熟悉它，并应用于工作，这其实就是因为你有如何使用AI去解决问题的能。后端也一样，
语言和框架只是工具，区别就是语法不同，然后不同的语言，再有一些自己特色，基本上就这些。解决问题的能力，理解的原理、积累的经验、培养的工程化思维这些才是重要的，所以说，你会一个语言的后端，去学习其它语言的后端，也是非常快的


加油，未来的后端工程师 🐍🚀