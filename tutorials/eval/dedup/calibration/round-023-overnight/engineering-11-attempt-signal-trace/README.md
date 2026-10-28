# Main-attempt direction signals

This separate diagnostic reads every available main attempt, including failed
retries. It extracts explicit direction signals from native lexical JSON and,
for diagnostic signals only, complete bare JSON. It never validates a reply,
changes a decision, or grants agreement credit. Fixed references and scores
stay unchanged. Whole-pair alternatives use the native maximum-match rule.
Unlabeled diagnostics receive no correctness comparison; unknown values stay
unknown. Counts are attempt signals, including repeats, rather than accuracy.

The original complete traces and HTTP bodies remain external. The checked-in
projections contain only recognized direction signals, native attempt status,
frozen-reference comparisons and receipt hashes. They contain no raw response
body or source text. Two actual failed-attempt checks confirm that otherwise
matching raw signals still leave their native rows invalid and uncredited.

# 主判断尝试的方向信号

这一单独诊断读取全部可用main尝试，包括失败重试。通过原生词法JSON提取方向，
也仅为诊断信号解码完整裸JSON；从不将回复判为有效、改变判断或授予一致率分数。
冻结参考和分数保持不变，整对备选沿用原生最大匹配方式。未标注诊断不作正确性
比较，未知值仍为未知。计数含尝试和重复，不代表准确率。

完整追踪和HTTP正文保留在外部。仓库投影只含识别到的方向信号、原生尝试状态、
冻结参考比较和凭据哈希，没有原始回复正文或原文。两项真实失败尝试核验确认，
即使原始信号与参考吻合，原生行仍保持无效且不得正确分数。
