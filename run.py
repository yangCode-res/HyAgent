from Agents.Entity_extraction.index import main
from ChatLLM.index import ChatLLM
from Logger.index import get_global_logger
if __name__ == "__main__":
    logger = get_global_logger()
    logger.info("Starting HyGraph...")
    main()
    logger.info("HyGraph finished.")
    logger.info("="*100)