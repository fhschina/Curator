# V0.6.2.25 完整命题配对实验

依据`.23`在线与`.24`离线审计，在任何新模型输出前声明本次改动。
当前目标仍未达成；这是固定main的局部critic诊断，不是生产注册或全量评价。

## 要检验的假设

词级diff会把一句话拆成若干shared/unique片段。critic需要把片段还原成完整命题，
核对对侧同义内容，再判UNCOVERED；按钮、导航指针、常识联想都不能代替完整命题比较。
H0347/H0453漏看B的“继续浏览即同意”，H0748把Stories指针当正文，H0186忽略实际profile主体，
H0333把具体申请方法当成topic已隐含；这些是开发来源，不把它们伪装成holdout。

## 隔离变量

- control：`.23`原selection system/pair/YAML及原schema展示顺序。
- coverage（候选）：新的完整命题提示与先说明、后status的schema展示顺序。
- 两臂均用`.24`已经离线验证的同一scope后处理规则：双向覆盖/明确保留损失不被taxonomy否决；
  MAIN_CONTENT单向损失与scope矛盾仍U。所有原确定路由保留。
- 两臂都输出同一dedup-retained-coverage-v2字段、枚举及约束。新版本不增加隐藏自由字段，
  不从解释反推并重写status；仅改变候选schema属性/required展示顺序，JSON字段顺序不是有效性条件。
  schema顺序与实际渲染消息均冻结，旧schema与配置文件不改。
- 复用原严格selection编译、非pruning raw解析、原文对齐及`.24`探针算法；
  后处理reason code保留其OFFLINE_SCOPE_POLICY名称作为算法来源，而非声称模型是离线生成。
  新行明确记录fresh请求摘要、实际attempt及scope_postprocessor。旧响应不成为新缓存。

scope本身不是新的实验变量，不能把两臂共同的代码修复当作候选prompt收益。
本轮control不同于旧`.14`control，报告须明确其为`.23`selection加相同scope后处理。
旧`.14/.23`分数仅历史背景，不能替代同步对照。

## 候选规则与成对保护

先识别两侧实际保留主体，再核对每个unique span对应的完整句义。
coverage_explanation短写“本侧命题/元素；对侧对应命题及位置；保留或未保留原因”，不输出长内部辩论。
源选中一个own unique span只是定位，不限定义务/主体的语义上下文为该span里的几个词。
对侧没有明确反向命题时用NO_EQUIVALENT_FOUND而非CONTRADICTS，不臆造opt-in、已点击、订单状态等。

采用通用成对示例，不含review ID、原站名、原姓名、原邮件或reference：

1. 同义“继续浏览即同意”加Accept按钮仍等价；明确“点击之前不使用”才是新条件。
2. 同站说明加Stories/author指针不自动扩展；private profile标题中的不同姓名绑定访问对象，优先身份冲突。
3. 同一招聘核心加联系人/邮件主题是具体方法，不由“申请岗位”推导；同方法完整翻译不算扩展。
4. 政策目的/数据用途/访问门槛/实际状态等真实损失仍保留；不因共用banner或无矛盾而一律等价。
5. 共用导航与单侧实际listing仍无非空共同记录；同记录缺字段可包含，值冲突不可包含。

忠实翻译语义material NONE；但翻译外观不使明确缺失条款自动等价。
H0158/H0313现有reference仍照常计分，可能的默认状态条款争议不自动改标签。

## 运行及准入

同258对、原分层权重、固定`.14`main、同两轮顺序；两臂各147个实际调用、111个确定输出。
R1 control→coverage，R2 coverage→control；1032公共输出，588基础模型请求，另16预检请求。
同已授权NVIDIA27B，temperature0/top_p1/4096/无thinking/timeout600/max_retries2/并发16。
逐对检查实际渲染预算input+4096+2048≤32768，不裁剪、不跳过、不改变模型。

新根`/raid/hfang/ihb/runs/v0.6.2.25-proposition-diagnostic`，自有Ray目录`/raid/hfang/ihb/r25diag`。
先两臂first/retry四个完整本地Ray→HTTP→writer→bind用例，再冻结代码、资源、schema顺序、call sets和消息摘要。
预检取R1各臂实际called前8个；两臂8/8且零native/outer纠错、HTTP200各8才进正式。
任何终态cell或确定性边界失败停止后续，不重启已观察run，不择优、不删失败pair。

完整schedule后才比较，两轮全部保留；加权primary各不低于同步control、over-group不增加、
两轮primary翻转不多于control。继承所有local_gates、38/22负例、20负例/3benign底线、4containment、
12exact、54translation与三个目标，最终schema/evidence完成100%，两臂native+outer并集/实际called≤1%。
HTTP429/500/503等请求重发另列，不以111个确定输出稀释重试分母。

只有上述全通过才支持fresh完整Judge局部测试，之后完整1000仍需75/75/79、over-group≤66及原分类保护。
然后独立人工400 holdout，最后20000；不得改reference、声称AI盲审、调用较大模型或自动release/commit/push。
