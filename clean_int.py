import logging,datetime
import argparse
import os
import traceback
from tqdm import tqdm
from datetime import datetime

ap = argparse.ArgumentParser(description="Script to clean ServiceNow interactions")
ap.add_argument("-d", "--debug", action="store_true", help="Set debug mode")
ap.add_argument("DumpFile", type=str, help="Text file with dump of interactions")
args = ap.parse_args()

is_debug = args.debug

log_level = logging.DEBUG if is_debug else logging.INFO
logging.basicConfig(format='%(asctime)s::%(levelname)s::%(message)s', level=log_level)

dump_file = args.DumpFile

keywords = ["System:", "Virtual Agent:", "Merci d'avoir utilisé le support Tchat bioMérieux",
            "le support Tchat", "Bienvenue au Support IS bioMérieux", "Bienvenue au Support",
            "Comment puis-je vous aider", "How may I help you"]

logging.info("Start")

try:
    cleaned_lines = []

    with open(dump_file,'r') as file:
        total_lines = sum(1 for _ in file)
    logging.info("Total lines to process in file "+dump_file+": "+str(total_lines))
    with open(dump_file,'r') as file:
        for line in tqdm(file, total=total_lines, desc='Processing lines'):
            line = line.strip()
            if len(line) >= 2 and not any(keyword in line for keyword in keywords):
                if line.startswith("[") or line.startswith("\""):
                    cleaned_line = line.split(":", 2)[-1].strip()
                    if len(cleaned_line) >= 2:
                        cleaned_lines.append(cleaned_line)
                else:
                    cleaned_lines.append(line)
    logging.info("Saving results")
    
    # Generate current date and time
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    cleaned_file_name = f"interaction_cleaned_{current_datetime}.txt"
    
    with open(cleaned_file_name, 'w',encoding='utf-8') as clean_file:
        clean_file.write('\n'.join(cleaned_lines))

    logging.info("End")

except Exception as err:
    logging.error(err)
    print(traceback.format_exc())
