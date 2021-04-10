# BreakingBERT@IITK at SemEval-2021

Task 9 : Statement Verification and Evidence Finding with Tables


Recently, there has been an interest in factual
verification and prediction over structured data
like tables and graphs. To circumvent any false
news incident, it is necessary to not only model
and predict over structured data efficiently but
also to explain those predictions. In this paper, as part of the SemEval-2021 Task 9, we
tackle the problem of fact verification and evidence finding over tabular data. There are two
subtasks. Given a table and a statement/fact,
subtask A determines whether the statement is
inferred from the tabular data, and subtask B
determines which cells in the table provide evidence for the former subtask. We make a comparison of the baselines and state-of-the-art approaches over the given SemTabFact dataset.
We also propose a novel approach CellBERT
to solve evidence finding as a form of the Natural Language Inference task. We obtain a 3-
way F1 score of 0.69 on subtask A and an F1
score of 0.65 on subtask B.

