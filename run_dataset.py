import argparse
import os
import json
import random
import torch
import numpy as np
import traceback
import psutil
from tqdm import tqdm
from datasets import load_dataset
from utils import open_config, create_model
from detector.attn import AttentionDetector
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from logging_utils import setup_logging, log_gpu_memory
import logging

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def log_system_info():
    """Log system information"""
    logging.info("=== SYSTEM INFO ===")
    logging.info(f"CPU count: {psutil.cpu_count()}")
    logging.info(f"RAM: {psutil.virtual_memory().total / 1024**3:.2f}GB total, {psutil.virtual_memory().available / 1024**3:.2f}GB available")
    
    if torch.cuda.is_available():
        logging.info(f"CUDA available: {torch.cuda.device_count()} GPU(s)")
        for i in range(torch.cuda.device_count()):
            logging.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        logging.info("CUDA not available")
    
    log_gpu_memory()

def main(args):
    # Setup logging
    logger, log_file = setup_logging(f"debug_{args.model_name}_{args.dataset_name.replace('/', '_')}.log")
    logging.info("=== SCRIPT STARTED ===")
    logging.info(f"Model: {args.model_name}")
    logging.info(f"Dataset: {args.dataset_name}")
    logging.info(f"Seed: {args.seed}")
    logging.info(f"Log file: {log_file}")
    
    try:
        log_system_info()
        
        set_seed(args.seed)
        logging.info("Seed set successfully")

        output_logs = f"./result/{args.dataset_name}/{args.model_name}-{args.seed}.json"
        output_result = f"./result/{args.dataset_name}/result.jsonl"
        dataset_output_file = f"./results/{args.model_name}_dataset.txt"
        
        logging.info(f"Output files configured: {dataset_output_file}")
        
        # Load model config
        logging.info("Loading model configuration...")
        model_config_path = f"./configs/model_configs/{args.model_name}_config.json"
        model_config = open_config(config_path=model_config_path)
        logging.info(f"Model config loaded from: {model_config_path}")
        
        # Create model
        logging.info("Creating model...")
        log_gpu_memory()
        model = create_model(config=model_config)
        logging.info("Model created successfully")
        log_gpu_memory()
        
        model.print_model_info()
        logging.info("Model info printed")

        # Load dataset - đây là nơi có thể bị dừng
        logging.info(f"Loading dataset: {args.dataset_name}")
        logging.info("This may take a while for first-time download...")
        
        try:
            dataset = load_dataset(args.dataset_name, trust_remote_code=True)
            logging.info("Dataset loaded successfully")
            logging.info(f"Dataset keys: {list(dataset.keys())}")
        except Exception as e:
            logging.error(f"Failed to load dataset: {str(e)}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            raise
        
        test_data = dataset['test']
        logging.info(f"Test data loaded: {len(test_data)} samples")
        
        # Create detector
        logging.info("Creating detector...")
        detector = AttentionDetector(model)
        logging.info(f"Detector created: {detector.name}")
        print("===================")
        print(f"Using detector: {detector.name}")
        print(f"Writing results to: {dataset_output_file}")

        labels, predictions, scores = [], [], []
        logs = []

        logging.info("Starting processing loop...")
        for idx, data in enumerate(tqdm(test_data), start=1):
            logging.debug(f"Processing sample {idx}")
            
            try:
                set_seed(args.seed)
                result = detector.detect(data['text'])
                detect = result[0]
                score = result[1]['focus_score']
                generated_text = result[1]['generated_text']
                
                logging.debug(f"Sample {idx} processed: detect={detect}, score={score}")

                output_one_line = generated_text.replace("\n", " ").strip()
                print(f"=== Sample {idx} ===")
                print(f"Input: {data['text']}")
                print(f"Output: {output_one_line}")
                print(f"Score: {score}")
                print(f"Detected Injection: {detect}")
                print("Check: ")
                print()

                # Write results
                with open(dataset_output_file, "a", encoding="utf-8") as f:
                    f.write(f"=== Sample {idx} ===\n")
                    f.write(f"Input: {data['text']}\n")
                    f.write(f"Output: {output_one_line}\n")
                    f.write(f"Score: {score}\n")
                    f.write(f"Detected Injection: {detect}\n")
                    f.write("Check: ")
                    f.write("\n")
                
                # Memory cleanup
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                labels.append(data['label'])
                predictions.append(detect)
                scores.append(1-score)

                result_data = {
                    "text": data['text'],
                    "label": data['label'],
                    "result": result
                }
                logs.append(result_data)
                
                # Log every 10 samples
                if idx % 10 == 0:
                    logging.info(f"Processed {idx} samples")
                    log_gpu_memory()
                    
            except Exception as e:
                logging.error(f"Error processing sample {idx}: {str(e)}")
                logging.error(f"Sample data: {data}")
                logging.error(f"Traceback: {traceback.format_exc()}")
                # Có thể continue hoặc raise tùy vào yêu cầu
                continue

        # Calculate metrics
        logging.info("Calculating metrics...")
        auc_score = roc_auc_score(labels, scores)
        auprc_score = average_precision_score(labels, scores)

        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        auc_score = round(auc_score, 3)
        auprc_score = round(auprc_score, 3)
        fnr = round(fnr, 3)
        fpr = round(fpr, 3)

        logging.info(f"Final results - AUC: {auc_score}, AUPRC: {auprc_score}, FNR: {fnr}, FPR: {fpr}")
        print(f"AUC Score: {auc_score}; AUPRC Score: {auprc_score}; FNR: {fnr}; FPR: {fpr}")
        
        # Save results
        os.makedirs(os.path.dirname(output_logs), exist_ok=True)
        with open(output_logs, "w") as f_out:
            f_out.write(json.dumps({"result": logs}, indent=4))

        os.makedirs(os.path.dirname(output_result), exist_ok=True)
        with open(output_result, "a") as f_out:
            f_out.write(json.dumps({
                "model": args.model_name,
                "seed": args.seed,
                "auc": auc_score,
                "auprc": auprc_score,
                "fnr": fnr,
                "fpr": fpr
            }) + "\n")
            
        logging.info("=== SCRIPT COMPLETED SUCCESSFULLY ===")
        
    except Exception as e:
        logging.error(f"CRITICAL ERROR: {str(e)}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prompt Injection Detection Script")
    
    parser.add_argument("--model_name", type=str, default="qwen2-attn",
                        help="Path to the model configuration file.")
    parser.add_argument("--dataset_name", type=str, default="deepset/prompt-injections", 
                        help="Path to the dataset.")
    parser.add_argument("--seed", type=int, default=0)
    
    args = parser.parse_args()

    main(args)