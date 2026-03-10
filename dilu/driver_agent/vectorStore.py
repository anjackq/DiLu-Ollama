import os
import textwrap
from rich import print
import json

# UPDATED IMPORTS: Adapting to LangChain v0.1+
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from dilu.scenario.envScenario import EnvScenario


class DrivingMemory:

    def __init__(self, encode_type='sce_language', db_path=None) -> None:
        self.encode_type = encode_type

        if encode_type == 'sce_encode':
            raise ValueError("encode_type sce_encode is deprecated for now.")

        elif encode_type == 'sce_language':
            api_type = os.environ.get("OPENAI_API_TYPE")
            # --- 1. Handle Azure ---
            if api_type == 'azure':
                self.embedding = OpenAIEmbeddings(
                    deployment=os.environ['AZURE_EMBED_DEPLOY_NAME'],
                    chunk_size=1
                )

            # --- 2. Handle OpenAI & Ollama ---
            elif api_type == 'openai':
                base_url = os.environ.get("OPENAI_API_BASE", "")

                # Check if we are running locally (Ollama)
                if "localhost" in base_url or "127.0.0.1" in base_url:
                    # CRITICAL: Ollama needs the specific model name.
                    # Default 'text-embedding-ada-002' will fail locally.
                    # We reuse the chat model (e.g. qwen2.5:14b) for embeddings.
                    model_name = os.environ.get("OPENAI_CHAT_MODEL", "qwen2.5:14b")

                    self.embedding = OpenAIEmbeddings(
                        model=model_name,
                        openai_api_base=base_url,
                        openai_api_key="ollama",  # Dummy key
                        check_embedding_ctx_length=False  # Prevents errors with local models
                    )
                    print(f"[yellow]Using Local Ollama Embeddings with model: {model_name}[/yellow]")
                else:
                    # Standard OpenAI (uses default ada-002)
                    self.embedding = OpenAIEmbeddings()

            # --- ADDED OLLAMA SUPPORT ---
            elif api_type == 'ollama':
                model_name = os.getenv('OLLAMA_EMBED_MODEL')
                print(f"[green]Using Ollama Embeddings[/green] with model: {model_name}")
                # Note: We use the OpenAI-compatible endpoint provided by Ollama
                self.embedding = OpenAIEmbeddings(
                    model=model_name,
                    openai_api_base=os.getenv("OLLAMA_API_BASE"),  # http://localhost:11434/v1
                    openai_api_key=os.getenv("OLLAMA_API_KEY"),  # 'ollama'
                    check_embedding_ctx_length=False  # Necessary for some local models
                )
            elif api_type == "gemini":
                openai_embed_base = str(os.getenv("OPENAI_API_BASE", "")).strip()
                openai_embed_key = str(os.getenv("OPENAI_API_KEY", "")).strip()
                if openai_embed_base and openai_embed_key:
                    model_name = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
                    print(
                        f"[green]Gemini mode:[/green] using OpenAI-compatible embeddings "
                        f"at {openai_embed_base} with model: {model_name}"
                    )
                    self.embedding = OpenAIEmbeddings(
                        model=model_name,
                        openai_api_base=openai_embed_base,
                        openai_api_key=openai_embed_key,
                        check_embedding_ctx_length=False
                    )
                else:
                    ollama_base = str(os.getenv("OLLAMA_API_BASE", "")).strip()
                    ollama_key = str(os.getenv("OLLAMA_API_KEY", "")).strip()
                    ollama_model = str(os.getenv("OLLAMA_EMBED_MODEL", "")).strip()
                    if ollama_base and ollama_key and ollama_model:
                        print(
                            f"[green]Gemini mode:[/green] using Ollama embeddings "
                            f"with model: {ollama_model}"
                        )
                        self.embedding = OpenAIEmbeddings(
                            model=ollama_model,
                            openai_api_base=ollama_base,
                            openai_api_key=ollama_key,
                            check_embedding_ctx_length=False
                        )
                    else:
                        raise ValueError(
                            "Gemini mode requires an embedding backend. Set OPENAI_API_BASE + OPENAI_API_KEY "
                            "(and optional OPENAI_EMBED_MODEL), or set OLLAMA_API_BASE + OLLAMA_API_KEY + OLLAMA_EMBED_MODEL."
                        )

            else:
                raise ValueError("Unknown OPENAI_API_TYPE: should be azure, openai, ollama, or gemini")

            # Define DB path
            db_path = os.path.join('./db', 'chroma_5_shot_20_mem/') if db_path is None else db_path

            # Initialize Chroma
            self.scenario_memory = Chroma(
                embedding_function=self.embedding,
                persist_directory=db_path
            )
        else:
            raise ValueError("Unknown ENCODE_TYPE: should be sce_encode or sce_language")

        # Safety check for collection
        try:
            count = self.scenario_memory._collection.count()
            print("==========Loaded ", db_path, " Memory. Total items: ", count, "==========")
        except Exception as e:
            print(f"[red]Warning loading memory count: {e}[/red]")

    def retriveMemory(self, driving_scenario: EnvScenario, frame_id: int, top_k: int = 5):
        if self.encode_type == 'sce_encode':
            pass
        elif self.encode_type == 'sce_language':
            query_scenario = driving_scenario.describe(frame_id)
            # Perform similarity search
            similarity_results = self.scenario_memory.similarity_search_with_score(
                query_scenario, k=top_k)
            fewshot_results = []
            for res, score in similarity_results:
                fewshot_results.append(res.metadata)
            return fewshot_results
        return []

    def addMemory(self, sce_descrip: str, human_question: str, response: str, action: int, sce: EnvScenario = None,
                  comments: str = ""):
        if self.encode_type == 'sce_language':
            sce_descrip = sce_descrip.replace("'", '')

        # Access raw collection to check for duplicates
        get_results = self.scenario_memory._collection.get(
            where_document={
                "$contains": sce_descrip
            }
        )

        if len(get_results['ids']) > 0:
            # Update existing memory
            id = get_results['ids'][0]
            self.scenario_memory._collection.update(
                ids=id,
                metadatas={"human_question": human_question,
                           'LLM_response': response, 'action': action, 'comments': comments}
            )
            print("Modify a memory item.")
        else:
            # Add new memory
            doc = Document(
                page_content=sce_descrip,
                metadata={"human_question": human_question,
                          'LLM_response': response, 'action': action, 'comments': comments}
            )
            self.scenario_memory.add_documents([doc])
            print("Add a memory item.")

    def deleteMemory(self, ids):
        self.scenario_memory.delete(ids=ids)
        print("Delete", len(ids), "memory items.")

    def combineMemory(self, other_memory):
        other_documents = other_memory.scenario_memory._collection.get(
            include=['documents', 'metadatas', 'embeddings'])
        current_documents = self.scenario_memory._collection.get(
            include=['documents', 'metadatas', 'embeddings'])

        def _safe_signature(doc, metadata):
            return (doc or "") + "||" + json.dumps(metadata or {}, sort_keys=True)

        current_ids = set(current_documents.get('ids') or [])
        current_sigs = set()
        current_docs = current_documents.get('documents') or []
        current_meta = current_documents.get('metadatas') or []
        for i in range(min(len(current_docs), len(current_meta))):
            current_sigs.add(_safe_signature(current_docs[i], current_meta[i]))

        other_embeddings = other_documents.get('embeddings') or []
        if hasattr(other_embeddings, 'tolist'):
            other_embeddings = other_embeddings.tolist()
        other_ids = other_documents.get('ids') or []
        other_docs = other_documents.get('documents') or []
        other_meta = other_documents.get('metadatas') or []

        merged = 0
        skipped = 0
        for i in range(len(other_embeddings)):
            candidate_id = other_ids[i]
            candidate_sig = _safe_signature(other_docs[i], other_meta[i])
            if candidate_id in current_ids or candidate_sig in current_sigs:
                skipped += 1
                continue

            self.scenario_memory._collection.add(
                embeddings=[other_embeddings[i]],  # Must be a list
                metadatas=[other_meta[i]],
                documents=[other_docs[i]],
                ids=[candidate_id]
            )
            current_ids.add(candidate_id)
            current_sigs.add(candidate_sig)
            merged += 1

        # Safe count check
        count = self.scenario_memory._collection.count()
        print("Merge complete. Added", merged, "items, skipped", skipped, "duplicates. Now the database has", count, "items.")


if __name__ == "__main__":
    pass
