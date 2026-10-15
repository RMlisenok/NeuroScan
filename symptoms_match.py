import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from class_info import CLASS_INFO, get_all_keywords, get_all_descriptions


def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s\-]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _keyword_hits(complaints_clean: str, keywords: list[str]) -> tuple[int, list[str]]:
    hits = []
    for kw in keywords:
        if kw.lower() in complaints_clean:
            hits.append(kw)
    return len(hits), hits


def analyze_complaints(complaints: str, predicted_class: str | None = None) -> dict:
    result = {
        "similarity": {},
        "keyword_hits": {},
        "matched": {},
        "best_match": None,
        "agreement": None,
        "warning": None,
        "has_text": False,
        "method": "tfidf+keywords",
    }

    cleaned = _clean_text(complaints)
    if len(cleaned) < 10:
        result["warning"] = "Жалобы не заполнены или слишком короткие для анализа."
        return result

    result["has_text"] = True
    classes = list(CLASS_INFO.keys())

    descriptions = get_all_descriptions()
    corpus = [cleaned] + [descriptions[c] for c in classes]
    vectorizer = TfidfVectorizer(
        lowercase=True, ngram_range=(1, 2), min_df=1, sublinear_tf=True,
    )
    tfidf = vectorizer.fit_transform(corpus)
    sims = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
    result["similarity"] = {cls: round(float(s), 3) for cls, s in zip(classes, sims)}

    all_kw = get_all_keywords()
    keyword_hits = {}
    matched = {}
    for cls in classes:
        n, hits = _keyword_hits(cleaned, all_kw.get(cls, []))
        keyword_hits[cls] = n
        matched[cls] = hits
    result["keyword_hits"] = keyword_hits
    result["matched"] = matched

    max_hits = max(keyword_hits.values()) or 1
    combined = {}
    for cls in classes:
        kw_score = keyword_hits[cls] / max_hits
        tfidf_score = result["similarity"][cls]
        combined[cls] = 0.6 * kw_score + 0.4 * tfidf_score
    result["combined"] = {cls: round(v, 3) for cls, v in combined.items()}

    best_match = max(combined, key=combined.get)
    result["best_match"] = best_match

    if predicted_class:
        result["agreement"] = (best_match == predicted_class)
        if best_match == predicted_class:
            result["warning"] = None
        else:
            pred_ru = CLASS_INFO.get(predicted_class, {}).get("name_ru", predicted_class)
            best_ru = CLASS_INFO.get(best_match, {}).get("name_ru", best_match)
            top_hits = matched.get(best_match, [])[:5]
            hint = f" (совпадения: {', '.join(top_hits)})" if top_hits else ""
            result["warning"] = (
                f"Внимание: жалобы пациента больше соответствуют классу "
                f"«{best_ru}»{hint}, тогда как модель определила «{pred_ru}». "
                f"Рекомендуется клиническая верификация."
            )

    return result