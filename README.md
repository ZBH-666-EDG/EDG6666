# 跌倒监测微服务 v2.0

基于 YOLOv8-pose + InsightFace + DeepSeek AI 的老年人跌倒检测微服务。

## 快速启动

```bash
pip install -r requirements.txt
python app.py
# 或双击 run.bat
```

访问 http://localhost:5001

### 环境变量（可选，启用 AI 分析）

```bash
cp .env.example .env
# 编辑 .env，填入 DeepSeek API Key
```

## 项目结构

```
├── app.py              # 主服务（Flask + SocketIO + 多线程）
├── config.py           # AI 配置
├── requirements.txt
├── run.bat             # Windows 一键启动
├── .env.example        # API Key 模板
├── init_db.py          # 数据库初始化
├── templates/
│   ├── index.html      # 监控主页
│   ├── register.html   # 人脸注册（三步向导）
│   ├── manage.html     # 用户管理
│   ├── history.html    # 跌倒历史
│   └── test.html       # 视频测试
└── static/
    ├── falls/          # 跌倒截图
    └── uploads/        # 注册照片
```

## 外部集成指南

### 1. iframe 嵌入视频流

```html
<iframe src="http://localhost:5001" width="800" height="600"></iframe>
<!-- 或仅嵌入视频画面 -->
<img src="http://localhost:5001/video_feed" width="640" height="480">
```

### 2. WebSocket 接收警报

```javascript
const ws = new WebSocket('ws://localhost:5001/ws');
ws.onmessage = function(e) {
    const alert = JSON.parse(e.data);
    // alert.type: 'warning' (level 1, 姿态不稳) or 'fall' (level 2, 跌倒确认)
    // alert.level: 1 or 2
    // alert.name: 识别的老人姓名
    // alert.confidence: 跌倒概率 0~1
    // alert.timestamp: 时间戳
};
```

### 3. SSE 接收警报（备选）

```javascript
const es = new EventSource('http://localhost:5001/events');
es.onmessage = function(e) {
    const alert = JSON.parse(e.data);
    // 格式同 WebSocket
};
```

### 4. REST API

| 方法 | 路由 | 说明 |
|------|------|------|
| GET | `/api/health` | 服务状态 `{"status":"ok","fps":15.3,"persons":1,"name":"张三"}` |
| GET | `/api/events?limit=20` | 事件列表 JSON |
| GET | `/api/events/<id>` | 单条事件详情（含 AI 报告） |
| POST | `/api/register_face` | 注册人脸 `{"name":"李四","image_base64":"..."}` |
| GET | `/api/faces` | 已注册人员列表 |
| DELETE | `/api/faces/<id>` | 删除人员 |
| PUT | `/api/faces/<id>/photo` | 添加面部照片（multipart） |

### 5. 注册人脸 API 示例

```bash
curl -X POST http://localhost:5001/api/register_face \
  -H "Content-Type: application/json" \
  -d '{"name":"李四","image_base64":"'$(base64 photo.jpg)'"}'
```

### 6. 健康检查

```bash
curl http://localhost:5001/api/health
# {"status":"ok","fps":25.3,"persons":1,"name":"张三","p_fall":0.12,"mode":"camera","ai_enabled":true}
```

## 多级警报

| 等级 | 触发条件 | 推送字段 | 前端表现 |
|------|----------|----------|----------|
| Level 1 (警告) | P_FALL 0.55~0.75 持续 | `{"level":1,"type":"warning","message":"姿态不稳"}` | 黄色闪烁遮罩 + 短促提示音 |
| Level 2 (跌倒) | P_FALL >= 0.75 连续2帧 | `{"level":2,"type":"fall","name":"张三","confidence":0.85}` | 红色遮罩 + 连续蜂鸣警报 |

## 线程模型

```
摄像头采集线程 → frame_queue → 检测线程(YOLO+人脸+跌倒)
                                    ↓
                         SSE推送 + WebSocket广播
                                    ↓
                    AI分析线程池(ThreadPoolExecutor, max=2)
```
