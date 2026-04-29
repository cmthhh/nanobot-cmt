# little changes

本文记录当前分支里对 nanobot 做的两个主要改动：历史检索索引，以及针对本地小模型上下文污染的会话清理命令。

## 1. 历史索引

### 问题

原项目的长期历史主要写在：

```text
.nanobot/workspace/memory/history.jsonl
```

这个文件适合作为主存储，因为它简单、追加写入、容易恢复。但随着长期使用，历史会跨越很多天甚至几个月，直接靠 grep 或逐行扫描检索会越来越慢。

这会影响两类场景：

- 用户想查找过去某段对话或决策。
- Dream 在整理长期记忆时，需要从旧历史里找出和当前批次相关的内容。

如果每次都全量扫描 `history.jsonl`，数据越多，耗时越明显。

### 功能

新增一个轻量 SQLite FTS 索引：

```text
.nanobot/workspace/memory/history_index.db
.nanobot/workspace/memory/.index_cursor
```

设计原则是：

- `history.jsonl` 仍然是唯一主存储。
- `history_index.db` 只是可重建的旁路索引。
- 索引不在每次对话后实时更新，而是在 Dream 等记忆维护流程中批量更新。
- 如果索引文件损坏或丢失，可以从 `history.jsonl` 重建。

这样避免了每轮对话都增加写入成本，同时让跨月历史检索更快。

### 实现位置

主要代码在：

```text
nanobot/agent/memory.py
```

核心对象：

```text
HistoryIndex
MemoryStore.sync_history_index()
MemoryStore.search_history()
```

Dream 开始运行时会先同步索引，然后可以用索引查找和当前批次相关的旧历史。

### 效果

索引带来的收益主要体现在历史变长以后：

- 小规模历史：差别不明显。
- 几千到几万条历史：FTS 查询会比全文件扫描更稳定。
- 长期个人助理场景：可以减少 Dream 每次查旧历史的成本。

它不是替代 `history.jsonl`，而是给长期历史加了一层搜索加速。

## 2. `/checknow` 会话污染清理

### 问题

在实际测试中发现如果使用本地小模型实现这一助手功能，尤其是 7B 级别模型，在 nanobot 的 agent 场景里容易出现上下文污染。

直接和模型聊天时，输入通常很短，模型表现正常。但通过 nanobot 时，模型会看到：

- 系统提示词
- 工具说明
- 最近会话
- 长期记忆
- 技能说明
- 历史摘要

上下文变长以后，小模型可能输出明显乱码，例如：

```text
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```

或混杂代码碎片、多语言碎片、重复短语。

更麻烦的是，这些坏回复会被写进：

```text
.nanobot/workspace/sessions/cli_direct.jsonl
```

后续每次对话又会把它们带回 prompt，模型就可能继续被污染。

手动编辑 `cli_direct.jsonl` 也不理想，因为 nanobot 运行时会把 session 缓存在内存里。运行中改磁盘文件，下一次保存可能被内存里的旧 session 覆盖。如果手动还是删改，需要在停止nanobot后修改保存再重新打开才可以。

### 功能

新增内置命令：

```text
/checknow
```

它会在 nanobot 运行中直接读取当前 session 的内存对象，检查最近 5 轮对话，删除明显乱码的 assistant 回复，然后调用项目自己的 session 保存流程写回文件。

删除前会自动备份当前 session 文件，例如：

```text
.nanobot/workspace/sessions/cli_direct.checknow-YYYYMMDD-HHMMSS.jsonl.bak
```

### 判断逻辑

`/checknow` 不是简单字符串替换，而是两层判断：

1. 调用模型，让模型保守判断最近 5 轮里哪些 assistant 消息明显乱码。
2. 再用规则二次确认，只删除非常明显的问题内容。

规则包括：

- 大量重复标点，例如连续很多 `!`
- 重复短片段
- 明显损坏字符
- 随机代码样碎片
- 高比例符号噪声

它默认只删除 assistant 消息，不删除用户消息。

### 实现位置

主要代码在：

```text
nanobot/command/builtin.py
```

命令注册在：

```text
register_builtin_commands()
```

并补了路由测试：

```text
tests/command/test_router_dispatchable.py
```

### 使用方式

在 nanobot 交互窗口输入：

```text
/checknow
```

它会返回清理结果，包括删除了几条消息、原因，以及备份文件位置。

### 边界

`/checknow` 只处理当前 session 的最近 5 轮对话。

它不会自动清理：

- `memory/history.jsonl`
- `memory/MEMORY.md`
- `USER.md`
- `SOUL.md`
- 其他 channel 的 session

