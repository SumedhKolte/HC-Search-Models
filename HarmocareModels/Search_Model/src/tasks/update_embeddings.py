import logging
import asyncio
from datetime import datetime, timezone
from sqlalchemy import text
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from psycopg2.extras import execute_values

from src.data.database import Database  # Changed from 'db' to Database
from src.utils.constants import EMBEDDING_MODELS, ENTITY_CONFIGS

logger = logging.getLogger(__name__)
model = SentenceTransformer(EMBEDDING_MODELS['bge_small'])

class EmbeddingUpdater:
    def __init__(self):
        self.db = Database()  # Create database instance
        
    def generate_embedding(self, text):
        return model.encode(text).tolist()

    async def fetch_new_records(self, entity):
        config = ENTITY_CONFIGS[entity]
        query = text(f"""
        SELECT {config['id_column']}, {config['text_column']}
        FROM {config['table_name']}
        WHERE {config['embedding_column']} IS NULL
        """)
        try:
            async with self.db.get_session() as session:  # Use async session
                result = await session.execute(query)
                return [dict(row) for row in result]
        except Exception as e:
            logger.error(f"❌ Failed to fetch new records for {entity}: {e}")
            return []

    async def update_embeddings(self, entity):
        logger.info(f"🔁 Updating embeddings for entity: {entity}")
        config = ENTITY_CONFIGS[entity]
        
        records = await self.fetch_new_records(entity)
        if not records:
            logger.info(f"✅ No new updates for: {entity}")
            return

        batch_data = []
        for record in tqdm(records, desc="Batches"):
            # Check for cancellation
            try:
                await asyncio.sleep(0)  # Yield to event loop
            except asyncio.CancelledError:
                logger.info(f"🛑 Cancelling embedding updates for {entity}")
                raise

            text = record[config['text_column']]
            embedding = self.generate_embedding(text)
            embedding_str = str(embedding)

            batch_data.append((
                record[config['id_column']],
                embedding_str,
                text,
                datetime.now(timezone.utc)
            ))

        query = f"""
        UPDATE {config['table_name']} AS t SET
            {config['embedding_column']} = v.embedding::vector(384),
            {config['text_column']} = v.text,
            updated_at = v.updated_at
        FROM (VALUES %s) AS v(id, embedding, text, updated_at)
        WHERE t.{config['id_column']} = v.id
        """

        try:
            with self.db.engine.begin() as conn:
                with conn.connection.cursor() as cur:
                    execute_values(cur, query, batch_data, template="(%s, %s, %s, %s)")
            logger.info(f"✅ Updated {len(batch_data)} records for {entity}")
        except Exception as e:
            logger.error(f"❌ Failed to update {entity}: {e}")

    async def run(self):
        try:
            for entity in ENTITY_CONFIGS.keys():
                await self.update_embeddings(entity)
            logger.info("✅ Completed embedding updates")
        except asyncio.CancelledError:
            logger.info("🛑 Embedding updates cancelled")
            raise

async def run_embedding_updates():
    """Run embedding updates for all entities"""
    try:
        logger.info("Starting embedding updates")
        updater = EmbeddingUpdater()
        await updater.run()
    except Exception as e:
        logger.error(f"Failed to run embedding updates: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(run_embedding_updates())
