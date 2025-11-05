from Agents.Entity_extraction.index import main
from ChatLLM.index import ChatLLM
from Logger.index import get_global_logger
from Store.index import get_memory
if __name__ == "__main__":
    memory = get_memory()
    
    logger = get_global_logger()
    logger.info("Starting HyGraph...")
    main()
    logger.info("HyGraph finished.")
    logger.info("="*100)
    memory.dump_json("./snapshots")