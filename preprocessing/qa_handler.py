# preprocessing/qa_handler.py
from typing import List, Dict, Any
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import gc
from queue import Empty

def log_gpu_memory(worker_log_prefix: str, stage: str):
    if not torch.cuda.is_available():
        return
    try:
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 1024**2
        print(f"{worker_log_prefix} [MEM_LOG] {stage} | Allouée: {allocated:.2f} MB | Réservée: {reserved:.2f} MB | Total: {total:.2f} MB")
    except Exception as e:
        print(f"{worker_log_prefix} [MEM_LOG] Impossible de logger la mémoire: {e}")

def _create_qa_generation_prompt(text_segment: str, title: str, entities: list, lang: str) -> str:
    if entities and isinstance(entities[0], tuple): entities_str = ", ".join([str(ent[0]) for ent in entities if ent and ent[0]])
    elif entities and isinstance(entities[0], str): entities_str = ", ".join(entities)
    else: entities_str = "aucune entité spécifique notable"
    max_segment_display_len = 1500
    text_segment_for_prompt_display = text_segment[:max_segment_display_len] + ("..." if len(text_segment) > max_segment_display_len else "")
    prompt = f"""Tu es un assistant expert en analyse de texte. Ta tâche est de générer une question, sa réponse, et un score de confiance à partir de l'extrait de texte fourni.
    L'extrait provient de l'article intitulé "{title}" et mentionne potentiellement les entités : [{entities_str}].

    Extrait de texte à analyser :
    ---
    {text_segment_for_prompt_display}
    ---

    Instructions STRICTES pour la génération :
    1.  **Question :** Doit commencer EXACTEMENT par "Question: ". Génère UNE question concise et pertinente à laquelle l'extrait répond. Si l'extrait ne permet pas de formuler une question claire et une réponse factuelle, écris "Question: N/A". NE PAS NUMÉROTER cette ligne.
    2.  **Réponse :** Doit commencer EXACTEMENT par "Réponse: ". Fournis la réponse directe et concise, basée UNIQUEMENT sur l'extrait. Si la Question est N/A, écris "Réponse: N/A". NE PAS NUMÉROTER cette ligne.
    3.  **Confiance\_LLM :** Doit commencer EXACTEMENT par "Confiance\_LLM: ". Fournis un score flottant entre 0.0 et 1.0. Si la Question est N/A, écris "Confiance\_LLM: 0.0". NE PAS NUMÉROTER cette ligne. NE RIEN AJOUTER sur cette ligne après le score numérique.
    4.  **Langue :** Il faut générer la question et la réponse dans cette langue : {lang}

    Voici un EXEMPLE de format de sortie PARFAITEMENT RESPECTÉ :
    Question: Quelle est la capitale de la France ?
    Réponse: Paris.
    Confiance_LLM: 0.9

    Ta production doit suivre IMPÉRATIVEMENT ce format, chaque item sur une nouvelle ligne distincte :
    """
    return prompt

def _parse_llm_qa_output(llm_output: str) -> dict:
    question, answer, llm_confidence = None, None, 0.0
    cleaned_output = llm_output.strip()
    q_match = re.search(r"^Question:\s*(.*)", cleaned_output, re.MULTILINE | re.IGNORECASE)
    a_match = re.search(r"^Réponse:\s*(.*)", cleaned_output, re.MULTILINE | re.IGNORECASE)
    c_match = re.search(r"^Confiance_LLM:\s*([0-1](?:\.[0-9]+)?)", cleaned_output, re.MULTILINE | re.IGNORECASE)
    if q_match and q_match.group(1).strip().upper() != "N/A": question = q_match.group(1).strip()
    if a_match and a_match.group(1).strip().upper() != "N/A": answer = a_match.group(1).strip()
    if c_match:
        try: llm_confidence = float(c_match.group(1).strip())
        except: pass
    if question is None and answer is None: llm_confidence = 0.0
    return {"generated_question": question, "generated_answer": answer, "llm_confidence": llm_confidence}

def qa_generation_worker(
        tasks_queue: Any,
        results_queue: Any,
        stop_event: Any,
        worker_id: int,
        gpu_id: int,
        llm_model_name: str,
        qa_params: Dict
):
    device = f"cuda:{gpu_id}"
    worker_log_prefix = f"[WORKER {worker_id} | GPU {gpu_id}]"
    model, tokenizer = None, None
    try:
        log_gpu_memory(worker_log_prefix, "Début du worker")
        tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(llm_model_name, torch_dtype=torch.float16, device_map=device, trust_remote_code=True)
        log_gpu_memory(worker_log_prefix, "Après chargement modèle")
        try:
            model = torch.compile(model, mode="max-autotune")
            log_gpu_memory(worker_log_prefix, "Après compilation")
        except Exception:
            print(f"{worker_log_prefix} [AVERTISSEMENT] torch.compile() a échoué.")
        while not stop_event.is_set():
            try:
                current_chunk_data = tasks_queue.get(timeout=1.0)
            except Empty:
                continue
            if current_chunk_data is None: break
            try:
                log_gpu_memory(worker_log_prefix, f"Début lot (taille {len(current_chunk_data)})")
                prompts = [_create_qa_generation_prompt(s.get('text',''), s.get('title',''), s.get('entities',[]), s.get('lang','fr')) for s in current_chunk_data]
                if not prompts:
                    results_queue.put(current_chunk_data)
                    continue
                inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1800).to(device)
                with torch.no_grad():
                    generated_ids = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_new_tokens=qa_params['max_new_tokens_qa'], pad_token_id=tokenizer.pad_token_id)
                decoded_outputs = tokenizer.batch_decode(generated_ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                parsed_qas = [_parse_llm_qa_output(out) for out in decoded_outputs]
                processed_segments = [{**seg, **pqa} for seg, pqa in zip(current_chunk_data, parsed_qas)]
                results_queue.put(processed_segments)
                del inputs, generated_ids, decoded_outputs
                log_gpu_memory(worker_log_prefix, "Fin lot")
            except torch.cuda.OutOfMemoryError as e:
                print(f"{worker_log_prefix} [ERREUR OOM] {e}")
                log_gpu_memory(worker_log_prefix, "Au moment de l'OOM")
                print(f"{worker_log_prefix} Remise du lot dans la file et nettoyage.")
                tasks_queue.put(current_chunk_data)
                del prompts
                if 'inputs' in locals(): del inputs
                gc.collect()
                torch.cuda.empty_cache()
                log_gpu_memory(worker_log_prefix, "Après nettoyage OOM")
                time.sleep(5)
                continue
            except Exception as e:
                results_queue.put([{**s, "error_qab_generation": str(e)} for s in current_chunk_data])
    except torch.cuda.OutOfMemoryError as e:
        print(f"{worker_log_prefix} [ERREUR OOM FATALE] Chargement initial: {e}")
        stop_event.set()
    except Exception as e:
        print(f"{worker_log_prefix} [ERREUR FATALE] {e}")
        stop_event.set()
    finally:
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        log_gpu_memory(worker_log_prefix, "Fin du worker")