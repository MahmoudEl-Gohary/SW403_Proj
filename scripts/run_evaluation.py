"""
Evaluation Script for RAG Prototypes

This script runs all 30 evaluation questions against all 4 RAG prototypes (P1-P4)
and collects raw results for later analysis.

Usage:
    uv run python scripts/run_evaluation.py --zip data/Cancer_Detection.zip

The script will:
1. Index the codebase with all 4 chunking strategies
2. Run each question against each prototype
3. Collect answers, latency, and selfcheck scores
4. Output results to data/evaluation_results_raw.json
"""

import argparse
import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

# Configuration
API_BASE_URL = "http://localhost:8000"
STRATEGIES = ["function", "ast", "context", "graph"]
STRATEGY_NAMES = {
    "function": "P1_function",
    "ast": "P2_ast", 
    "context": "P3_context",
    "graph": "P4_graph"
}


async def wait_for_api(client: httpx.AsyncClient, timeout: int = 30) -> bool:
    """Wait for the API to be ready."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            response = await client.get(f"{API_BASE_URL}/health")
            if response.status_code == 200:
                print("✓ API is ready")
                return True
        except httpx.ConnectError:
            pass
        await asyncio.sleep(1)
    return False


async def index_codebase(
    client: httpx.AsyncClient, 
    zip_path: Path, 
    strategy: str
) -> dict[str, Any]:
    """Index the codebase with a specific strategy."""
    print(f"  Indexing with strategy: {strategy}...")
    
    # First, set the strategy
    await client.post(
        f"{API_BASE_URL}/config",
        json={"chunking_strategy": strategy}
    )
    
    # Upload and index the zip file
    with open(zip_path, "rb") as f:
        files = {"file": (zip_path.name, f, "application/zip")}
        response = await client.post(
            f"{API_BASE_URL}/index/upload",
            files=files,
            timeout=300.0  # 5 min timeout for large files
        )
    
    if response.status_code != 200:
        print(f"    ✗ Failed to index: {response.text}")
        return {"success": False, "error": response.text}
    
    result = response.json()
    print(f"    ✓ Indexed {result['num_documents']} files → {result['num_chunks']} chunks")
    print(f"    Collection: {result['collection_name']}")
    return result


async def query_prototype(
    client: httpx.AsyncClient,
    question: str,
    strategy: str,
    collection: str,
    k: int = 5
) -> dict[str, Any]:
    """Query a single prototype with a question."""
    start_time = time.time()
    
    try:
        response = await client.post(
            f"{API_BASE_URL}/query",
            json={
                "query": question,
                "strategy": strategy,
                "k": k,
                "collection": collection
            },
            timeout=120.0  # 2 min timeout
        )
        latency_ms = (time.time() - start_time) * 1000
        
        if response.status_code != 200:
            return {
                "answer": None,
                "error": response.text,
                "latency_ms": latency_ms,
                "num_chunks": 0,
                "retrieved_chunks": []
            }
        
        result = response.json()
        return {
            "answer": result["answer"],
            "latency_ms": latency_ms,
            "num_chunks": result["num_chunks"],
            "retrieved_chunks": [
                {
                    "content": chunk["content"][:500],  # Truncate for readability
                    "source": chunk["source"],
                    "name": chunk.get("name")
                }
                for chunk in result["retrieved_chunks"]
            ]
        }
    except Exception as e:
        return {
            "answer": None,
            "error": str(e),
            "latency_ms": (time.time() - start_time) * 1000,
            "num_chunks": 0,
            "retrieved_chunks": []
        }


async def run_selfcheck(
    client: httpx.AsyncClient,
    question: str,
    answer: str,
    collection: str,
    k: int = 3
) -> dict[str, Any]:
    """Run selfcheck on an answer to detect hallucinations."""
    if not answer:
        return {"similarity_score": 0.0, "is_hallucinating": True}
    
    try:
        response = await client.post(
            f"{API_BASE_URL}/selfcheck",
            json={
                "query": question,
                "response": answer,
                "collection": collection,
                "k": k
            },
            timeout=60.0
        )
        
        if response.status_code != 200:
            return {"similarity_score": 0.0, "is_hallucinating": True, "error": response.text}
        
        return response.json()
    except Exception as e:
        return {"similarity_score": 0.0, "is_hallucinating": True, "error": str(e)}


def get_category_name(question_id: int) -> tuple[str, int]:
    """Get category name and number from question ID."""
    if question_id <= 10:
        return "1_simple_lookup", 1
    elif question_id <= 20:
        return "2_local_context", 2
    else:
        return "3_global_relational", 3


async def run_evaluation(zip_path: Path, output_path: Path, skip_indexing: bool = False, strategies_filter: list[str] | None = None):
    """Main evaluation loop."""
    
    # Determine which strategies to run
    active_strategies = strategies_filter if strategies_filter else STRATEGIES
    
    # Load evaluation questions
    questions_path = Path(__file__).parent.parent / "data" / "evaluation_questions.json"
    with open(questions_path) as f:
        eval_data = json.load(f)
    
    # Flatten questions
    all_questions = []
    for category_key, category_data in eval_data["categories"].items():
        for q in category_data["questions"]:
            all_questions.append({
                **q,
                "category_key": category_key,
                "category_description": category_data["description"]
            })
    
    print(f"Loaded {len(all_questions)} evaluation questions")
    
    async with httpx.AsyncClient(timeout=None) as client:
        # Check API availability
        if not await wait_for_api(client):
            print("✗ API not available. Start it with: uv run uvicorn src.api.main:app")
            return
        
        # Index codebase with selected strategies
        collections = {}
        
        if not skip_indexing:
            print(f"\n=== Phase 1: Indexing Codebase ({len(active_strategies)} strategies) ===")
            for strategy in active_strategies:
                result = await index_codebase(client, zip_path, strategy)
                if result.get("success"):
                    collections[strategy] = result["collection_name"]
                else:
                    print(f"  ✗ Skipping {strategy} due to indexing failure")
        else:
            # Get existing collections
            print("\n=== Skipping indexing, using existing collections ===")
            response = await client.get(f"{API_BASE_URL}/databases")
            if response.status_code == 200:
                db_list = response.json()["databases"]
                for strategy in STRATEGIES:
                    # Find matching collection
                    for db in db_list:
                        if strategy in db.lower():
                            collections[strategy] = db
                            print(f"  Found collection for {strategy}: {db}")
                            break
        
        if not collections:
            print("✗ No collections available for evaluation")
            return
        
        # Run evaluation
        print("\n=== Phase 2: Running Evaluation ===")
        results = []
        
        for i, question in enumerate(all_questions, 1):
            print(f"\n[{i}/{len(all_questions)}] Q{question['id']}: {question['question'][:60]}...")
            
            category_name, category_num = get_category_name(question["id"])
            
            question_result = {
                "question_id": question["id"],
                "category": category_name,
                "category_number": category_num,
                "question": question["question"],
                "ground_truth": question["ground_truth"],
                "prototype_results": {}
            }
            
            for strategy, collection in collections.items():
                prototype_name = STRATEGY_NAMES[strategy]
                print(f"  → {prototype_name}...", end=" ", flush=True)
                
                # Query
                query_result = await query_prototype(
                    client, question["question"], strategy, collection
                )
                
                # Selfcheck (only if we got an answer)
                selfcheck_result = {"similarity_score": None, "is_hallucinating": None}
                if query_result.get("answer"):
                    selfcheck_result = await run_selfcheck(
                        client, question["question"], query_result["answer"], collection
                    )
                
                question_result["prototype_results"][prototype_name] = {
                    "answer": query_result.get("answer"),
                    "retrieved_chunks": query_result.get("retrieved_chunks", []),
                    "latency_ms": round(query_result.get("latency_ms", 0), 2),
                    "num_chunks": query_result.get("num_chunks", 0),
                    "selfcheck_score": selfcheck_result.get("similarity_score"),
                    "hallucination_detected": selfcheck_result.get("is_hallucinating"),
                    # to be assessed manually
                    "evaluation": {
                        "answered_correctly": None,
                        "partial_answer": None,
                        "hallucination_type": None,
                        "notes": None
                    }
                }
                
                status = "✓" if query_result.get("answer") else "✗"
                latency = query_result.get("latency_ms", 0)
                print(f"{status} ({latency:.0f}ms)")
            
            results.append(question_result)
        
        # Build output
        output = {
            "metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "codebase": zip_path.stem,
                "total_questions": len(all_questions),
                "strategies_tested": list(collections.keys()),
                "collections": collections,
                "api_base_url": API_BASE_URL
            },
            "results": results,
            "summary": {
                "by_prototype": {},
                "by_category": {}
            }
        }
        
        # Generate summary statistics
        for strategy in STRATEGIES:
            prototype_name = STRATEGY_NAMES[strategy]
            if strategy not in collections:
                continue
                
            proto_stats = {
                "total": len(results),
                "answered": 0,
                "hallucinations_detected": 0,
                "avg_latency_ms": 0,
                "by_category": {}
            }
            
            latencies = []
            for cat_num in [1, 2, 3]:
                proto_stats["by_category"][f"category_{cat_num}"] = {"answered": 0, "total": 0}
            
            for result in results:
                proto_result = result["prototype_results"].get(prototype_name, {})
                cat_num = result["category_number"]
                
                proto_stats["by_category"][f"category_{cat_num}"]["total"] += 1
                
                if proto_result.get("answer"):
                    proto_stats["answered"] += 1
                    proto_stats["by_category"][f"category_{cat_num}"]["answered"] += 1
                
                if proto_result.get("hallucination_detected"):
                    proto_stats["hallucinations_detected"] += 1
                
                if proto_result.get("latency_ms"):
                    latencies.append(proto_result["latency_ms"])
            
            proto_stats["avg_latency_ms"] = round(sum(latencies) / len(latencies), 2) if latencies else 0
            output["summary"]["by_prototype"][prototype_name] = proto_stats
        
        # Save results
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        
        print(f"\n=== Evaluation Complete ===")
        print(f"Results saved to: {output_path}")
        print(f"\nSummary:")
        for proto, stats in output["summary"]["by_prototype"].items():
            print(f"  {proto}: {stats['answered']}/{stats['total']} answered, "
                  f"{stats['hallucinations_detected']} hallucinations detected, "
                  f"avg latency: {stats['avg_latency_ms']:.0f}ms")


def main():
    parser = argparse.ArgumentParser(description="Run RAG prototype evaluation")
    parser.add_argument(
        "--zip", 
        type=Path, 
        default=Path("data/Cancer_Detection.zip"),
        help="Path to the codebase zip file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/evaluation_results_raw.json"),
        help="Output path for results JSON"
    )
    parser.add_argument(
        "--skip-indexing",
        action="store_true",
        help="Skip indexing and use existing collections"
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default=None,
        help="Comma-separated list of strategies to run (e.g., 'context,graph' for P3/P4 only)"
    )
    args = parser.parse_args()
    
    if not args.zip.exists():
        print(f"✗ Zip file not found: {args.zip}")
        return
    
    # Parse strategies if provided
    strategies = None
    if args.strategies:
        strategies = [s.strip() for s in args.strategies.split(",")]
        print(f"Running with strategies: {strategies}")
    
    asyncio.run(run_evaluation(args.zip, args.output, args.skip_indexing, strategies))


if __name__ == "__main__":
    main()

