import sacrebleu
from rouge_score import rouge_scorer

def evaluate_nlg_metrics(generated_texts, reference_texts):
    """
    Evaluate language quality using BLEU and ROUGE metrics.

    :param generated_texts: list of strings (model outputs)
    :param reference_texts: list of strings (ground truth references)
    :return: dict with BLEU and ROUGE metrics
    """
    # --- 1) BLEU with sacrebleu ---

    bleu = sacrebleu.corpus_bleu(
        generated_texts,
        [reference_texts],
        lowercase=True
    )
    bleu_score = bleu.score

    # --- 2) ROUGE ---
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1_scores, rouge2_scores, rougeL_scores = [], [], []

    for ref, hyp in zip(reference_texts, generated_texts):
        scores = scorer.score(ref, hyp)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)

    avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
    avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores)
    avg_rougeL = sum(rougeL_scores) / len(rougeL_scores)

    return {
        "BLEU": bleu_score,
        "ROUGE-1": avg_rouge1,
        "ROUGE-2": avg_rouge2,
        "ROUGE-L": avg_rougeL
    }


if __name__ == "__main__":
    generated = [
        "heart size within normal limits no focal consolidation",
        "there is a right lower lobe pneumonia with small pleural effusion"
    ]
    reference = [
        "the heart size is normal no consolidation is seen",
        "right lower lobe pneumonia is evident with a small pleural effusion"
    ]

    results = evaluate_nlg_metrics(generated, reference)
    print("NLG Metrics:", results)