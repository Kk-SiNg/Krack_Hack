import cloudscraper
from bs4 import BeautifulSoup
import json
import time
import re
import os

HEADERS = {"User-Agent": "Mozilla/5.0"}
TOPIC_URLS = [
    # Sorting
    "https://www.geeksforgeeks.org/bubble-sort-algorithm/",
    "https://www.geeksforgeeks.org/selection-sort-algorithm-2/",
    "https://www.geeksforgeeks.org/insertion-sort-algorithm/",
    "https://www.geeksforgeeks.org/merge-sort/",
    "https://www.geeksforgeeks.org/quick-sort-algorithm/",
    "https://www.geeksforgeeks.org/heap-sort/",
    "https://www.geeksforgeeks.org/counting-sort/",
    "https://www.geeksforgeeks.org/radix-sort/",

    # Searching
    "https://www.geeksforgeeks.org/binary-search/",
    "https://www.geeksforgeeks.org/linear-search/",
    "https://www.geeksforgeeks.org/ternary-search/",

    # Graph
    "https://www.geeksforgeeks.org/breadth-first-search-or-bfs-for-a-graph/",
    "https://www.geeksforgeeks.org/depth-first-search-or-dfs-for-a-graph/",
    "https://www.geeksforgeeks.org/dijkstras-shortest-path-algorithm-greedy-algo-7/",
    "https://www.geeksforgeeks.org/bellman-ford-algorithm-dp-23/",
    "https://www.geeksforgeeks.org/floyd-warshall-algorithm-dp-16/",
    "https://www.geeksforgeeks.org/prims-minimum-spanning-tree-mst-greedy-algo-5/",
    "https://www.geeksforgeeks.org/kruskals-minimum-spanning-tree-algorithm-greedy-algo-2/",
    "https://www.geeksforgeeks.org/topological-sorting/",
    "https://www.geeksforgeeks.org/detect-cycle-in-a-graph/",
    "https://www.geeksforgeeks.org/detect-cycle-undirected-graph/",
    "https://www.geeksforgeeks.org/bipartite-graph/",
    "https://www.geeksforgeeks.org/strongly-connected-components/",
    "https://www.geeksforgeeks.org/articulation-points-or-cut-vertices-in-a-graph/",
    "https://www.geeksforgeeks.org/bridge-in-a-graph/",
    "https://www.geeksforgeeks.org/tarjan-algorithm-find-strongly-connected-components/",

    # DP
    "https://www.geeksforgeeks.org/0-1-knapsack-problem-dp-10/",
    "https://www.geeksforgeeks.org/longest-common-subsequence-dp-4/",
    "https://www.geeksforgeeks.org/longest-increasing-subsequence-dp-3/",
    "https://www.geeksforgeeks.org/coin-change-dp-7/",
    "https://www.geeksforgeeks.org/matrix-chain-multiplication-dp-8/",
    "https://www.geeksforgeeks.org/edit-distance-dp-5/",
    "https://www.geeksforgeeks.org/subset-sum-problem-dp-25/",
    "https://www.geeksforgeeks.org/rod-cutting-dp-13/",
    "https://www.geeksforgeeks.org/palindrome-partitioning-dp-17/",
    "https://www.geeksforgeeks.org/word-break-problem-dp-32/",
    "https://www.geeksforgeeks.org/egg-dropping-puzzle-dp-11/",
    "https://www.geeksforgeeks.org/maximum-size-sub-matrix-with-all-1s-in-a-binary-matrix/",
    "https://www.geeksforgeeks.org/longest-palindromic-subsequence-dp-12/",
    "https://www.geeksforgeeks.org/boolean-parenthesization-problem-dp-37/",

    # Trees
    "https://www.geeksforgeeks.org/tree-traversals-inorder-preorder-and-postorder/",
    "https://www.geeksforgeeks.org/level-order-tree-traversal/",
    "https://www.geeksforgeeks.org/binary-search-tree-set-1-search-and-insertion/",
    "https://www.geeksforgeeks.org/binary-search-tree-set-2-delete/",
    "https://www.geeksforgeeks.org/avl-tree-set-1-insertion/",
    "https://www.geeksforgeeks.org/segment-tree-set-1-sum-of-given-range/",
    "https://www.geeksforgeeks.org/diameter-of-a-binary-tree/",
    "https://www.geeksforgeeks.org/lowest-common-ancestor-binary-tree-set-1/",
    "https://www.geeksforgeeks.org/check-for-balanced-tree/",
    "https://www.geeksforgeeks.org/determine-if-two-trees-are-identical/",
    "https://www.geeksforgeeks.org/maximum-width-of-a-binary-tree/",
    "https://www.geeksforgeeks.org/find-the-minimum-element-in-a-binary-search-tree/",
    "https://www.geeksforgeeks.org/inorder-predecessor-successor-given-key-bst/",
    "https://www.geeksforgeeks.org/check-if-a-binary-tree-is-bst-simple-and-efficient-approach/",
    "https://www.geeksforgeeks.org/kth-smallest-element-in-bst-using-o1-extra-space/",
    "https://www.geeksforgeeks.org/red-black-tree-set-1-introduction-2/",

    # Linked List
    "https://www.geeksforgeeks.org/reverse-a-linked-list/",
    "https://www.geeksforgeeks.org/detect-loop-in-a-linked-list/",
    "https://www.geeksforgeeks.org/merge-two-sorted-linked-lists/",
    "https://www.geeksforgeeks.org/find-middle-of-singly-linked-list/",
    "https://www.geeksforgeeks.org/flattening-a-linked-list/",
    "https://www.geeksforgeeks.org/add-two-numbers-represented-by-linked-lists/",
    "https://www.geeksforgeeks.org/intersection-of-two-linked-lists/",
    "https://www.geeksforgeeks.org/remove-nth-node-from-end-of-the-linked-list/",
    "https://www.geeksforgeeks.org/clone-linked-list-next-random-pointer-o1-space/",

    # Recursion / Backtracking
    "https://www.geeksforgeeks.org/n-queen-problem-backtracking-3/",
    "https://www.geeksforgeeks.org/rat-in-a-maze-problem-when-movement-in-all-possible-directions-is-allowed/",
    "https://www.geeksforgeeks.org/sudoku-backtracking-7/",
    "https://www.geeksforgeeks.org/the-knights-tour-problem/",
    "https://www.geeksforgeeks.org/tower-of-hanoi-algorithm/",
    "https://www.geeksforgeeks.org/m-coloring-problem-backtracking-5/",
    "https://www.geeksforgeeks.org/print-all-permutations-of-a-string-in-java/",

    # Number Theory & Bit Magic
    "https://www.geeksforgeeks.org/sieve-of-eratosthenes/",
    "https://www.geeksforgeeks.org/euclidean-algorithms-basic-and-extended/",
    "https://www.geeksforgeeks.org/modular-exponentiation-power-in-modular-arithmetic/",
    "https://www.geeksforgeeks.org/fibonacci-number/",
    "https://www.geeksforgeeks.org/count-set-bits-in-an-integer/",
    "https://www.geeksforgeeks.org/find-the-element-that-appears-once/",
    "https://www.geeksforgeeks.org/program-to-find-whether-a-no-is-power-of-two/",

    # Arrays
    "https://www.geeksforgeeks.org/program-for-array-rotation-continued-reversal-algorithm/",
    "https://www.geeksforgeeks.org/largest-sum-contiguous-subarray/",
    "https://www.geeksforgeeks.org/trapping-rain-water/",
    "https://www.geeksforgeeks.org/chocolate-distribution-problem/",
    "https://www.geeksforgeeks.org/find-the-missing-number/",
    "https://www.geeksforgeeks.org/merge-two-sorted-arrays/",
    "https://www.geeksforgeeks.org/moores-voting-algorithm/",
    "https://www.geeksforgeeks.org/stock-buy-sell/",

    # String
    "https://www.geeksforgeeks.org/kmp-algorithm-for-pattern-searching/",
    "https://www.geeksforgeeks.org/rabin-karp-algorithm-for-pattern-searching/",
    "https://www.geeksforgeeks.org/z-algorithm-linear-time-pattern-searching-algorithm/",
    "https://www.geeksforgeeks.org/reverse-a-string-in-c-cpp-different-methods/",
    "https://www.geeksforgeeks.org/check-whether-two-strings-are-anagram-of-each-other/",
    "https://www.geeksforgeeks.org/longest-palindrome-substring-set-1/",
    "https://www.geeksforgeeks.org/longest-common-prefix-using-sorting/",
    "https://www.geeksforgeeks.org/wildcard-pattern-matching/",

    # Stack / Queue
    "https://www.geeksforgeeks.org/next-greater-element/",
    "https://www.geeksforgeeks.org/the-stock-span-problem/",
    "https://www.geeksforgeeks.org/largest-rectangle-under-histogram/",
    "https://www.geeksforgeeks.org/queue-using-stacks/",
    "https://www.geeksforgeeks.org/lru-cache-implementation/",
    "https://www.geeksforgeeks.org/sliding-window-maximum-maximum-of-all-subarrays-of-size-k/",
    "https://www.geeksforgeeks.org/valid-expression-with-valid-parentheses/",
    "https://www.geeksforgeeks.org/design-a-stack-that-supports-getmin-in-o1-time-and-o1-extra-space/",
    "https://www.geeksforgeeks.org/circular-queue-set-1-introduction-array-implementation/",

    # Greedy & Hashing & Misc
    "https://www.geeksforgeeks.org/activity-selection-problem-greedy-algo-1/",
    "https://www.geeksforgeeks.org/job-sequencing-problem/",
    "https://www.geeksforgeeks.org/huffman-coding-greedy-algo-3/",
    "https://www.geeksforgeeks.org/fractional-knapsack-problem/",
    "https://www.geeksforgeeks.org/two-pointers-technique/",
    "https://www.geeksforgeeks.org/window-sliding-technique/",
    "https://www.geeksforgeeks.org/union-find-algorithm-set-2-union-by-rank/",
    "https://www.geeksforgeeks.org/find-itinerary-from-a-given-list-of-tickets/",
    "https://www.geeksforgeeks.org/count-distinct-elements-in-every-window-of-size-k/",
    "https://www.geeksforgeeks.org/find-whether-an-array-is-subset-of-another-array-set-1/",
    "https://www.geeksforgeeks.org/trie-insert-and-search/",
    "https://www.geeksforgeeks.org/binary-heap/"
]

scraper = cloudscraper.create_scraper()

def get_complexity_weight(url):
    if "-dp-" in url or "dynamic-programming" in url: return 3.0
    elif "graph" in url or "tree" in url or "backtracking" in url: return 2.0
    return 1.0

def detect_language(code):
    cpp_signals = ["#include", "iostream", "int main", "cout", "cin", "vector<", "std::"]
    py_signals = ["def ", "print(", "import ", "range(", "input(", "self.", "len("]
    code_lower = code.lower()
    
    if sum(1 for s in cpp_signals if s in code_lower) > sum(1 for s in py_signals if s in code_lower): return "cpp"
    elif sum(1 for s in py_signals if s in code_lower) > sum(1 for s in cpp_signals if s in code_lower): return "python"
    return "unknown"

def remove_comments(code, lang):
    if lang == "cpp":
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        return '\n'.join([re.sub(r'(?<!:)//(?!/).*$', '', line).rstrip() for line in code.split('\n')])
    else:
        return '\n'.join([line.split('#')[0].rstrip() if '#' in line and '"' not in line and "'" not in line else line for line in code.split('\n')])

def extract_code_blocks(url):
    try:
        resp = scraper.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        print(f"  ❌ Failed: {url} — {e}")
        return []

    soup = BeautifulSoup(resp.text, "lxml")
    results = []
    
    code_containers = soup.select(".code-container")
    if not code_containers: code_containers = soup.select("pre, td.code")

    for container in code_containers:
        for gutter in container.select('.gutter, .numbering, .line-numbers'):
            gutter.decompose()

        raw_text = container.get_text() 
        junk = ["Copy", "Output", "Time Complexity", "Auxiliary Space", "Complexity Analysis"]
        
        clean_lines = [
            line.rstrip() for line in raw_text.split("\n") 
            if line.strip() and not any(line.strip().startswith(j) for j in junk)
        ]

        code_text = "\n".join(clean_lines)
        if len(code_text) < 20 or len(clean_lines) < 5 or len(clean_lines) > 200: continue

        lang = detect_language(code_text)
        if lang not in ("cpp", "python"): continue

        comment_char = "//" if lang == "cpp" else "#"
        has_comments = sum(1 for l in clean_lines if comment_char in l) >= 2 or '/*' in code_text
        if not has_comments: continue

        code_no_comments = remove_comments(code_text, lang)
        if len([l for l in code_no_comments.split("\n") if l.strip()]) < 3: continue

        results.append({
            "code_with_comments": code_text,
            "code_without_comments": code_no_comments,
            "has_comments": True,
            "language": lang,
            "source_url": url,
            "complexity_weight": get_complexity_weight(url)
        })
    return results

if __name__ == "__main__":
    os.makedirs("dataset_batches", exist_ok=True)
    training_pairs = []

    print(f"Scraping {len(TOPIC_URLS)} GFG articles... 🕷️\n")

    for i, url in enumerate(TOPIC_URLS):
        print(f"[{i+1}/{len(TOPIC_URLS)}] {url}")
        training_pairs.extend(extract_code_blocks(url))
        time.sleep(1.5)

    BATCH_SIZE = 50 
    total_batches = (len(training_pairs) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in range(total_batches):
        batch = training_pairs[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
        with open(f"dataset_batches/gfg_training_batch_{i+1}.json", "w", encoding="utf-8") as f:
            json.dump(batch, f, indent=2, ensure_ascii=False)

    print(f"\n✅ SCRAPING COMPLETE: {len(training_pairs)} pairs saved into {total_batches} batches. 📂")