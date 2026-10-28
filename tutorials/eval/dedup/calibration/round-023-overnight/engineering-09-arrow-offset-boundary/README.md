# Arrow/Pandas source-offset compatibility

The first exact-CORE index passed 998 direct-record renders but failed native
execution: Arrow/Pandas converts nullable integer span offsets to floating
values. Python string slices require integer offsets. Native early shutdown
preserved null outputs and missing inference receipts; the integrity boundary
stopped the run. Its original slot and all actual calls remain intact.

A real Arrow/Pandas conversion reproduces 18 old-template render failures on
the 44-row preflight panel. Explicit integer casts at string slice boundaries
allow all 44 to render. Two independent conditional branches preserve the native
AST depth limit of ten; failed depth-eleven drafts remain external. The revised
index passes 998 native render projections, including 30 with an index, and
proves every indexed nonwhitespace character was already visible in the
opposite original packet. Removing only the index restores each original
rendered prompt byte-for-byte. These are CPU checks with zero model calls.

The actual native xref1 evaluation failure is not repaired retrospectively.
Xref2 has a distinct frozen prompt hash and a separately predeclared group.
No native Python, model, sampling, visible window, parser or validator changes.
The live GPU outcome is tracked separately from the passing CPU proof.

# Arrow/Pandas字符偏移兼容性

首版精确CORE索引通过998次直接记录渲染，却在原生执行中失败：Arrow/Pandas将
带空值的整数偏移转换为浮点数，而Python字符串切片要求整数。原生提前停止保留了
空结果和缺失推理凭据，完整性检查随即停止运行。原运行及真实调用均予以保留。

真实Arrow/Pandas往返转换复现了44行预检中18行的旧模板渲染错误。仅在字符串切片
边界显式转换整数后，44行全部通过。将两种方向写成独立条件分支，使原生AST深度
保持在十以内；深度十一的失败草稿仍保留在外部目录。修订索引通过998次原生渲染，
其中30次包含索引，并逐项证明索引的非空白字符此前已在对侧原始数据包显示。
仅移除新增索引后原提示逐字节恢复。这些CPU检查没有模型调用。

不追溯修补xref1的真实评测失败。xref2使用不同冻结哈希和独立的预先声明运行组。
原生Python、模型、采样、窗口、解析器和校验器均未改变。实际GPU结果与CPU通过分开记录。
