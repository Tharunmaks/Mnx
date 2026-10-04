"""Classic programs written by hand in 10 languages.

Every implementation prints exactly `expected`; training/check_code.py runs
or compiles each one and compares the output, so the training data only
contains programs that really work. Also used for "now in <language>"
follow-ups.
"""

LANGS = {
    # key: (display name, file extension, how users name it)
    "python": ("Python", "py", ["Python", "python", "py"]),
    "javascript": ("JavaScript", "js", ["JavaScript", "JS", "javascript", "node"]),
    "typescript": ("TypeScript", "ts", ["TypeScript", "TS", "typescript"]),
    "c": ("C", "c", ["C", "c"]),
    "cpp": ("C++", "cpp", ["C++", "c++", "cpp"]),
    "java": ("Java", "java", ["Java", "java"]),
    "go": ("Go", "go", ["Go", "Golang", "go"]),
    "rust": ("Rust", "rs", ["Rust", "rust"]),
    "ruby": ("Ruby", "rb", ["Ruby", "ruby"]),
    "php": ("PHP", "php", ["PHP", "php"]),
}

RUN_HINT = {
    "python": "python {f}", "javascript": "node {f}", "typescript": "npx tsx {f}", "c": "gcc {f} -o app && ./app",
    "cpp": "g++ {f} -o app && ./app", "java": "javac {f} && java {cls}", "go": "go run {f}", "rust": "rustc {f} -o app && ./app",
    "ruby": "ruby {f}", "php": "php {f}",
}

ALGOS = {
    "fizzbuzz": {
        "title": "FizzBuzz", "cls": "FizzBuzz",
        "asks": ["Write FizzBuzz in {L}", "fizzbuzz {l}", "{L} program for FizzBuzz from 1 to 15", "Can you code FizzBuzz in {L}?"],
        "desc": "prints FizzBuzz for 1 to 15",
        "expected": "1\n2\nFizz\n4\nBuzz\nFizz\n7\n8\nFizz\nBuzz\n11\nFizz\n13\n14\nFizzBuzz\n",
        "impl": {
            "python": "for i in range(1, 16):\n    if i % 15 == 0:\n        print(\"FizzBuzz\")\n    elif i % 3 == 0:\n        print(\"Fizz\")\n    elif i % 5 == 0:\n        print(\"Buzz\")\n    else:\n        print(i)\n",
            "javascript": "for (let i = 1; i <= 15; i++) {\n  if (i % 15 === 0) console.log(\"FizzBuzz\");\n  else if (i % 3 === 0) console.log(\"Fizz\");\n  else if (i % 5 === 0) console.log(\"Buzz\");\n  else console.log(i);\n}\n",
            "typescript": "for (let i: number = 1; i <= 15; i++) {\n  let out: string = \"\";\n  if (i % 3 === 0) out += \"Fizz\";\n  if (i % 5 === 0) out += \"Buzz\";\n  console.log(out || String(i));\n}\n",
            "c": "#include <stdio.h>\n\nint main(void) {\n    for (int i = 1; i <= 15; i++) {\n        if (i % 15 == 0) printf(\"FizzBuzz\\n\");\n        else if (i % 3 == 0) printf(\"Fizz\\n\");\n        else if (i % 5 == 0) printf(\"Buzz\\n\");\n        else printf(\"%d\\n\", i);\n    }\n    return 0;\n}\n",
            "cpp": "#include <iostream>\n\nint main() {\n    for (int i = 1; i <= 15; i++) {\n        if (i % 15 == 0) std::cout << \"FizzBuzz\\n\";\n        else if (i % 3 == 0) std::cout << \"Fizz\\n\";\n        else if (i % 5 == 0) std::cout << \"Buzz\\n\";\n        else std::cout << i << '\\n';\n    }\n    return 0;\n}\n",
            "java": "public class FizzBuzz {\n    public static void main(String[] args) {\n        for (int i = 1; i <= 15; i++) {\n            if (i % 15 == 0) System.out.println(\"FizzBuzz\");\n            else if (i % 3 == 0) System.out.println(\"Fizz\");\n            else if (i % 5 == 0) System.out.println(\"Buzz\");\n            else System.out.println(i);\n        }\n    }\n}\n",
            "go": "package main\n\nimport \"fmt\"\n\nfunc main() {\n\tfor i := 1; i <= 15; i++ {\n\t\tswitch {\n\t\tcase i%15 == 0:\n\t\t\tfmt.Println(\"FizzBuzz\")\n\t\tcase i%3 == 0:\n\t\t\tfmt.Println(\"Fizz\")\n\t\tcase i%5 == 0:\n\t\t\tfmt.Println(\"Buzz\")\n\t\tdefault:\n\t\t\tfmt.Println(i)\n\t\t}\n\t}\n}\n",
            "rust": "fn main() {\n    for i in 1..=15 {\n        match (i % 3, i % 5) {\n            (0, 0) => println!(\"FizzBuzz\"),\n            (0, _) => println!(\"Fizz\"),\n            (_, 0) => println!(\"Buzz\"),\n            _ => println!(\"{}\", i),\n        }\n    }\n}\n",
            "ruby": "(1..15).each do |i|\n  if i % 15 == 0\n    puts \"FizzBuzz\"\n  elsif i % 3 == 0\n    puts \"Fizz\"\n  elsif i % 5 == 0\n    puts \"Buzz\"\n  else\n    puts i\n  end\nend\n",
            "php": "<?php\nfor ($i = 1; $i <= 15; $i++) {\n    if ($i % 15 == 0) echo \"FizzBuzz\\n\";\n    elseif ($i % 3 == 0) echo \"Fizz\\n\";\n    elseif ($i % 5 == 0) echo \"Buzz\\n\";\n    else echo $i . \"\\n\";\n}\n",
        },
    },
    "factorial": {
        "title": "Factorial", "cls": "Factorial",
        "asks": ["Write a factorial program in {L}", "factorial function in {l}", "{L} code to compute the factorial of a number", "How do I calculate factorial in {L}?"],
        "desc": "computes factorials (shows 10! = 3628800)",
        "expected": "10! = 3628800\n",
        "impl": {
            "python": "def factorial(n):\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result\n\n\nprint(f\"10! = {factorial(10)}\")\n",
            "javascript": "function factorial(n) {\n  let result = 1;\n  for (let i = 2; i <= n; i++) result *= i;\n  return result;\n}\n\nconsole.log(`10! = ${factorial(10)}`);\n",
            "typescript": "function factorial(n: number): number {\n  let result = 1;\n  for (let i = 2; i <= n; i++) result *= i;\n  return result;\n}\n\nconsole.log(`10! = ${factorial(10)}`);\n",
            "c": "#include <stdio.h>\n\nunsigned long long factorial(int n) {\n    unsigned long long result = 1;\n    for (int i = 2; i <= n; i++) result *= i;\n    return result;\n}\n\nint main(void) {\n    printf(\"10! = %llu\\n\", factorial(10));\n    return 0;\n}\n",
            "cpp": "#include <iostream>\n\nunsigned long long factorial(int n) {\n    unsigned long long result = 1;\n    for (int i = 2; i <= n; i++) result *= i;\n    return result;\n}\n\nint main() {\n    std::cout << \"10! = \" << factorial(10) << '\\n';\n    return 0;\n}\n",
            "java": "public class Factorial {\n    static long factorial(int n) {\n        long result = 1;\n        for (int i = 2; i <= n; i++) result *= i;\n        return result;\n    }\n\n    public static void main(String[] args) {\n        System.out.println(\"10! = \" + factorial(10));\n    }\n}\n",
            "go": "package main\n\nimport \"fmt\"\n\nfunc factorial(n int) uint64 {\n\tvar result uint64 = 1\n\tfor i := 2; i <= n; i++ {\n\t\tresult *= uint64(i)\n\t}\n\treturn result\n}\n\nfunc main() {\n\tfmt.Printf(\"10! = %d\\n\", factorial(10))\n}\n",
            "rust": "fn factorial(n: u64) -> u64 {\n    (1..=n).product()\n}\n\nfn main() {\n    println!(\"10! = {}\", factorial(10));\n}\n",
            "ruby": "def factorial(n)\n  (1..n).reduce(1, :*)\nend\n\nputs \"10! = #{factorial(10)}\"\n",
            "php": "<?php\nfunction factorial(int $n): int {\n    $result = 1;\n    for ($i = 2; $i <= $n; $i++) $result *= $i;\n    return $result;\n}\n\necho \"10! = \" . factorial(10) . \"\\n\";\n",
        },
    },
    "fibonacci": {
        "title": "Fibonacci", "cls": "Fibonacci",
        "asks": ["Write a Fibonacci program in {L}", "fibonacci series in {l}", "{L} code that prints the first 10 Fibonacci numbers", "Fibonacci sequence using {L}"],
        "desc": "prints the first 10 Fibonacci numbers",
        "expected": "0 1 1 2 3 5 8 13 21 34\n",
        "impl": {
            "python": "a, b = 0, 1\nnums = []\nfor _ in range(10):\n    nums.append(a)\n    a, b = b, a + b\nprint(\" \".join(map(str, nums)))\n",
            "javascript": "const nums = [];\nlet [a, b] = [0, 1];\nfor (let i = 0; i < 10; i++) {\n  nums.push(a);\n  [a, b] = [b, a + b];\n}\nconsole.log(nums.join(\" \"));\n",
            "typescript": "const nums: number[] = [];\nlet a = 0;\nlet b = 1;\nfor (let i = 0; i < 10; i++) {\n  nums.push(a);\n  [a, b] = [b, a + b];\n}\nconsole.log(nums.join(\" \"));\n",
            "c": "#include <stdio.h>\n\nint main(void) {\n    long a = 0, b = 1;\n    for (int i = 0; i < 10; i++) {\n        printf(i ? \" %ld\" : \"%ld\", a);\n        long next = a + b;\n        a = b;\n        b = next;\n    }\n    printf(\"\\n\");\n    return 0;\n}\n",
            "cpp": "#include <iostream>\n\nint main() {\n    long a = 0, b = 1;\n    for (int i = 0; i < 10; i++) {\n        if (i) std::cout << ' ';\n        std::cout << a;\n        long next = a + b;\n        a = b;\n        b = next;\n    }\n    std::cout << '\\n';\n    return 0;\n}\n",
            "java": "public class Fibonacci {\n    public static void main(String[] args) {\n        long a = 0, b = 1;\n        StringBuilder sb = new StringBuilder();\n        for (int i = 0; i < 10; i++) {\n            if (i > 0) sb.append(' ');\n            sb.append(a);\n            long next = a + b;\n            a = b;\n            b = next;\n        }\n        System.out.println(sb);\n    }\n}\n",
            "go": "package main\n\nimport (\n\t\"fmt\"\n\t\"strings\"\n)\n\nfunc main() {\n\ta, b := 0, 1\n\tparts := []string{}\n\tfor i := 0; i < 10; i++ {\n\t\tparts = append(parts, fmt.Sprint(a))\n\t\ta, b = b, a+b\n\t}\n\tfmt.Println(strings.Join(parts, \" \"))\n}\n",
            "rust": "fn main() {\n    let (mut a, mut b) = (0u64, 1u64);\n    let mut nums = Vec::new();\n    for _ in 0..10 {\n        nums.push(a.to_string());\n        let next = a + b;\n        a = b;\n        b = next;\n    }\n    println!(\"{}\", nums.join(\" \"));\n}\n",
            "ruby": "a, b = 0, 1\nnums = []\n10.times do\n  nums << a\n  a, b = b, a + b\nend\nputs nums.join(\" \")\n",
            "php": "<?php\n$a = 0;\n$b = 1;\n$nums = [];\nfor ($i = 0; $i < 10; $i++) {\n    $nums[] = $a;\n    [$a, $b] = [$b, $a + $b];\n}\necho implode(\" \", $nums) . \"\\n\";\n",
        },
    },
    "primes": {
        "title": "Prime numbers", "cls": "Primes",
        "asks": ["Write a {L} program to print prime numbers", "primes below 30 in {l}", "{L} function to check if a number is prime", "list prime numbers using {L}"],
        "desc": "checks numbers for primality and prints the primes below 30",
        "expected": "2 3 5 7 11 13 17 19 23 29\n",
        "impl": {
            "python": "def is_prime(n):\n    if n < 2:\n        return False\n    d = 2\n    while d * d <= n:\n        if n % d == 0:\n            return False\n        d += 1\n    return True\n\n\nprint(\" \".join(str(n) for n in range(30) if is_prime(n)))\n",
            "javascript": "function isPrime(n) {\n  if (n < 2) return false;\n  for (let d = 2; d * d <= n; d++) if (n % d === 0) return false;\n  return true;\n}\n\nconst primes = [];\nfor (let n = 0; n < 30; n++) if (isPrime(n)) primes.push(n);\nconsole.log(primes.join(\" \"));\n",
            "typescript": "function isPrime(n: number): boolean {\n  if (n < 2) return false;\n  for (let d = 2; d * d <= n; d++) if (n % d === 0) return false;\n  return true;\n}\n\nconst primes: number[] = [];\nfor (let n = 0; n < 30; n++) if (isPrime(n)) primes.push(n);\nconsole.log(primes.join(\" \"));\n",
            "c": "#include <stdio.h>\n\nint is_prime(int n) {\n    if (n < 2) return 0;\n    for (int d = 2; d * d <= n; d++)\n        if (n % d == 0) return 0;\n    return 1;\n}\n\nint main(void) {\n    int first = 1;\n    for (int n = 0; n < 30; n++) {\n        if (is_prime(n)) {\n            printf(first ? \"%d\" : \" %d\", n);\n            first = 0;\n        }\n    }\n    printf(\"\\n\");\n    return 0;\n}\n",
            "cpp": "#include <iostream>\n\nbool isPrime(int n) {\n    if (n < 2) return false;\n    for (int d = 2; d * d <= n; d++)\n        if (n % d == 0) return false;\n    return true;\n}\n\nint main() {\n    bool first = true;\n    for (int n = 0; n < 30; n++) {\n        if (isPrime(n)) {\n            if (!first) std::cout << ' ';\n            std::cout << n;\n            first = false;\n        }\n    }\n    std::cout << '\\n';\n    return 0;\n}\n",
            "java": "public class Primes {\n    static boolean isPrime(int n) {\n        if (n < 2) return false;\n        for (int d = 2; d * d <= n; d++)\n            if (n % d == 0) return false;\n        return true;\n    }\n\n    public static void main(String[] args) {\n        StringBuilder sb = new StringBuilder();\n        for (int n = 0; n < 30; n++) {\n            if (isPrime(n)) {\n                if (sb.length() > 0) sb.append(' ');\n                sb.append(n);\n            }\n        }\n        System.out.println(sb);\n    }\n}\n",
            "go": "package main\n\nimport (\n\t\"fmt\"\n\t\"strings\"\n)\n\nfunc isPrime(n int) bool {\n\tif n < 2 {\n\t\treturn false\n\t}\n\tfor d := 2; d*d <= n; d++ {\n\t\tif n%d == 0 {\n\t\t\treturn false\n\t\t}\n\t}\n\treturn true\n}\n\nfunc main() {\n\tprimes := []string{}\n\tfor n := 0; n < 30; n++ {\n\t\tif isPrime(n) {\n\t\t\tprimes = append(primes, fmt.Sprint(n))\n\t\t}\n\t}\n\tfmt.Println(strings.Join(primes, \" \"))\n}\n",
            "rust": "fn is_prime(n: u32) -> bool {\n    if n < 2 {\n        return false;\n    }\n    let mut d = 2;\n    while d * d <= n {\n        if n % d == 0 {\n            return false;\n        }\n        d += 1;\n    }\n    true\n}\n\nfn main() {\n    let primes: Vec<String> = (0..30).filter(|&n| is_prime(n)).map(|n| n.to_string()).collect();\n    println!(\"{}\", primes.join(\" \"));\n}\n",
            "ruby": "def prime?(n)\n  return false if n < 2\n  (2..Math.sqrt(n)).none? { |d| (n % d).zero? }\nend\n\nputs (0...30).select { |n| prime?(n) }.join(\" \")\n",
            "php": "<?php\nfunction isPrime(int $n): bool {\n    if ($n < 2) return false;\n    for ($d = 2; $d * $d <= $n; $d++) {\n        if ($n % $d == 0) return false;\n    }\n    return true;\n}\n\n$primes = array_filter(range(0, 29), 'isPrime');\necho implode(\" \", $primes) . \"\\n\";\n",
        },
    },
    "reverse": {
        "title": "Reverse a string", "cls": "ReverseString",
        "asks": ["Reverse a string in {L}", "{l} code to reverse a string", "Write a {L} function that reverses text", "string reversal program in {L}"],
        "desc": "reverses a string (\"hello world\" → \"dlrow olleh\")",
        "expected": "dlrow olleh\n",
        "impl": {
            "python": "def reverse(text):\n    return text[::-1]\n\n\nprint(reverse(\"hello world\"))\n",
            "javascript": "function reverse(text) {\n  return [...text].reverse().join(\"\");\n}\n\nconsole.log(reverse(\"hello world\"));\n",
            "typescript": "function reverse(text: string): string {\n  return [...text].reverse().join(\"\");\n}\n\nconsole.log(reverse(\"hello world\"));\n",
            "c": "#include <stdio.h>\n#include <string.h>\n\nvoid reverse(char *s) {\n    size_t n = strlen(s);\n    for (size_t i = 0; i < n / 2; i++) {\n        char t = s[i];\n        s[i] = s[n - 1 - i];\n        s[n - 1 - i] = t;\n    }\n}\n\nint main(void) {\n    char text[] = \"hello world\";\n    reverse(text);\n    printf(\"%s\\n\", text);\n    return 0;\n}\n",
            "cpp": "#include <algorithm>\n#include <iostream>\n#include <string>\n\nint main() {\n    std::string text = \"hello world\";\n    std::reverse(text.begin(), text.end());\n    std::cout << text << '\\n';\n    return 0;\n}\n",
            "java": "public class ReverseString {\n    public static void main(String[] args) {\n        String text = \"hello world\";\n        System.out.println(new StringBuilder(text).reverse());\n    }\n}\n",
            "go": "package main\n\nimport \"fmt\"\n\nfunc reverse(s string) string {\n\tr := []rune(s)\n\tfor i, j := 0, len(r)-1; i < j; i, j = i+1, j-1 {\n\t\tr[i], r[j] = r[j], r[i]\n\t}\n\treturn string(r)\n}\n\nfunc main() {\n\tfmt.Println(reverse(\"hello world\"))\n}\n",
            "rust": "fn main() {\n    let text = \"hello world\";\n    let reversed: String = text.chars().rev().collect();\n    println!(\"{}\", reversed);\n}\n",
            "ruby": "def reverse(text)\n  text.reverse\nend\n\nputs reverse(\"hello world\")\n",
            "php": "<?php\nfunction reverseText(string $text): string {\n    return strrev($text);\n}\n\necho reverseText(\"hello world\") . \"\\n\";\n",
        },
    },
    "bubble_sort": {
        "title": "Bubble sort", "cls": "BubbleSort",
        "asks": ["Implement bubble sort in {L}", "bubble sort {l}", "Write a {L} program to sort an array with bubble sort", "sorting algorithm in {L}: bubble sort"],
        "desc": "sorts an array with bubble sort ([5, 2, 9, 1, 5, 6] → 1 2 5 5 6 9)",
        "expected": "1 2 5 5 6 9\n",
        "impl": {
            "python": "def bubble_sort(items):\n    items = list(items)\n    n = len(items)\n    for i in range(n):\n        swapped = False\n        for j in range(n - 1 - i):\n            if items[j] > items[j + 1]:\n                items[j], items[j + 1] = items[j + 1], items[j]\n                swapped = True\n        if not swapped:\n            break\n    return items\n\n\nprint(\" \".join(map(str, bubble_sort([5, 2, 9, 1, 5, 6]))))\n",
            "javascript": "function bubbleSort(items) {\n  const a = [...items];\n  for (let i = 0; i < a.length; i++) {\n    let swapped = false;\n    for (let j = 0; j < a.length - 1 - i; j++) {\n      if (a[j] > a[j + 1]) {\n        [a[j], a[j + 1]] = [a[j + 1], a[j]];\n        swapped = true;\n      }\n    }\n    if (!swapped) break;\n  }\n  return a;\n}\n\nconsole.log(bubbleSort([5, 2, 9, 1, 5, 6]).join(\" \"));\n",
            "typescript": "function bubbleSort(items: number[]): number[] {\n  const a = [...items];\n  for (let i = 0; i < a.length; i++) {\n    let swapped = false;\n    for (let j = 0; j < a.length - 1 - i; j++) {\n      if (a[j] > a[j + 1]) {\n        [a[j], a[j + 1]] = [a[j + 1], a[j]];\n        swapped = true;\n      }\n    }\n    if (!swapped) break;\n  }\n  return a;\n}\n\nconsole.log(bubbleSort([5, 2, 9, 1, 5, 6]).join(\" \"));\n",
            "c": "#include <stdio.h>\n\nvoid bubble_sort(int *a, int n) {\n    for (int i = 0; i < n; i++) {\n        int swapped = 0;\n        for (int j = 0; j < n - 1 - i; j++) {\n            if (a[j] > a[j + 1]) {\n                int t = a[j];\n                a[j] = a[j + 1];\n                a[j + 1] = t;\n                swapped = 1;\n            }\n        }\n        if (!swapped) break;\n    }\n}\n\nint main(void) {\n    int a[] = {5, 2, 9, 1, 5, 6};\n    int n = sizeof a / sizeof a[0];\n    bubble_sort(a, n);\n    for (int i = 0; i < n; i++) printf(i ? \" %d\" : \"%d\", a[i]);\n    printf(\"\\n\");\n    return 0;\n}\n",
            "cpp": "#include <iostream>\n#include <utility>\n#include <vector>\n\nvoid bubbleSort(std::vector<int> &a) {\n    for (size_t i = 0; i < a.size(); i++) {\n        bool swapped = false;\n        for (size_t j = 0; j + 1 < a.size() - i; j++) {\n            if (a[j] > a[j + 1]) {\n                std::swap(a[j], a[j + 1]);\n                swapped = true;\n            }\n        }\n        if (!swapped) break;\n    }\n}\n\nint main() {\n    std::vector<int> a = {5, 2, 9, 1, 5, 6};\n    bubbleSort(a);\n    for (size_t i = 0; i < a.size(); i++) std::cout << (i ? \" \" : \"\") << a[i];\n    std::cout << '\\n';\n    return 0;\n}\n",
            "java": "public class BubbleSort {\n    static void bubbleSort(int[] a) {\n        for (int i = 0; i < a.length; i++) {\n            boolean swapped = false;\n            for (int j = 0; j < a.length - 1 - i; j++) {\n                if (a[j] > a[j + 1]) {\n                    int t = a[j];\n                    a[j] = a[j + 1];\n                    a[j + 1] = t;\n                    swapped = true;\n                }\n            }\n            if (!swapped) break;\n        }\n    }\n\n    public static void main(String[] args) {\n        int[] a = {5, 2, 9, 1, 5, 6};\n        bubbleSort(a);\n        StringBuilder sb = new StringBuilder();\n        for (int i = 0; i < a.length; i++) sb.append(i > 0 ? \" \" : \"\").append(a[i]);\n        System.out.println(sb);\n    }\n}\n",
            "go": "package main\n\nimport (\n\t\"fmt\"\n\t\"strings\"\n)\n\nfunc bubbleSort(a []int) {\n\tfor i := 0; i < len(a); i++ {\n\t\tswapped := false\n\t\tfor j := 0; j < len(a)-1-i; j++ {\n\t\t\tif a[j] > a[j+1] {\n\t\t\t\ta[j], a[j+1] = a[j+1], a[j]\n\t\t\t\tswapped = true\n\t\t\t}\n\t\t}\n\t\tif !swapped {\n\t\t\tbreak\n\t\t}\n\t}\n}\n\nfunc main() {\n\ta := []int{5, 2, 9, 1, 5, 6}\n\tbubbleSort(a)\n\tparts := make([]string, len(a))\n\tfor i, v := range a {\n\t\tparts[i] = fmt.Sprint(v)\n\t}\n\tfmt.Println(strings.Join(parts, \" \"))\n}\n",
            "rust": "fn bubble_sort(a: &mut Vec<i32>) {\n    let n = a.len();\n    for i in 0..n {\n        let mut swapped = false;\n        for j in 0..n - 1 - i {\n            if a[j] > a[j + 1] {\n                a.swap(j, j + 1);\n                swapped = true;\n            }\n        }\n        if !swapped {\n            break;\n        }\n    }\n}\n\nfn main() {\n    let mut a = vec![5, 2, 9, 1, 5, 6];\n    bubble_sort(&mut a);\n    let out: Vec<String> = a.iter().map(|v| v.to_string()).collect();\n    println!(\"{}\", out.join(\" \"));\n}\n",
            "ruby": "def bubble_sort(items)\n  a = items.dup\n  loop do\n    swapped = false\n    (a.length - 1).times do |j|\n      if a[j] > a[j + 1]\n        a[j], a[j + 1] = a[j + 1], a[j]\n        swapped = true\n      end\n    end\n    break unless swapped\n  end\n  a\nend\n\nputs bubble_sort([5, 2, 9, 1, 5, 6]).join(\" \")\n",
            "php": "<?php\nfunction bubbleSort(array $a): array {\n    $n = count($a);\n    for ($i = 0; $i < $n; $i++) {\n        $swapped = false;\n        for ($j = 0; $j < $n - 1 - $i; $j++) {\n            if ($a[$j] > $a[$j + 1]) {\n                [$a[$j], $a[$j + 1]] = [$a[$j + 1], $a[$j]];\n                $swapped = true;\n            }\n        }\n        if (!$swapped) break;\n    }\n    return $a;\n}\n\necho implode(\" \", bubbleSort([5, 2, 9, 1, 5, 6])) . \"\\n\";\n",
        },
    },
    "binary_search": {
        "title": "Binary search", "cls": "BinarySearch",
        "asks": ["Implement binary search in {L}", "binary search {l}", "Write a {L} program for binary search on a sorted array", "{L} code to search a sorted list quickly"],
        "desc": "finds an item in a sorted array with binary search",
        "expected": "Found 7 at index 3\n",
        "impl": {
            "python": "def binary_search(items, target):\n    lo, hi = 0, len(items) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if items[mid] == target:\n            return mid\n        if items[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\n\n\nprint(f\"Found 7 at index {binary_search([1, 3, 5, 7, 9, 11], 7)}\")\n",
            "javascript": "function binarySearch(items, target) {\n  let lo = 0;\n  let hi = items.length - 1;\n  while (lo <= hi) {\n    const mid = (lo + hi) >> 1;\n    if (items[mid] === target) return mid;\n    if (items[mid] < target) lo = mid + 1;\n    else hi = mid - 1;\n  }\n  return -1;\n}\n\nconsole.log(`Found 7 at index ${binarySearch([1, 3, 5, 7, 9, 11], 7)}`);\n",
            "typescript": "function binarySearch(items: number[], target: number): number {\n  let lo = 0;\n  let hi = items.length - 1;\n  while (lo <= hi) {\n    const mid = (lo + hi) >> 1;\n    if (items[mid] === target) return mid;\n    if (items[mid] < target) lo = mid + 1;\n    else hi = mid - 1;\n  }\n  return -1;\n}\n\nconsole.log(`Found 7 at index ${binarySearch([1, 3, 5, 7, 9, 11], 7)}`);\n",
            "c": "#include <stdio.h>\n\nint binary_search(const int *a, int n, int target) {\n    int lo = 0, hi = n - 1;\n    while (lo <= hi) {\n        int mid = lo + (hi - lo) / 2;\n        if (a[mid] == target) return mid;\n        if (a[mid] < target) lo = mid + 1;\n        else hi = mid - 1;\n    }\n    return -1;\n}\n\nint main(void) {\n    int a[] = {1, 3, 5, 7, 9, 11};\n    printf(\"Found 7 at index %d\\n\", binary_search(a, 6, 7));\n    return 0;\n}\n",
            "cpp": "#include <iostream>\n#include <vector>\n\nint binarySearch(const std::vector<int> &a, int target) {\n    int lo = 0, hi = static_cast<int>(a.size()) - 1;\n    while (lo <= hi) {\n        int mid = lo + (hi - lo) / 2;\n        if (a[mid] == target) return mid;\n        if (a[mid] < target) lo = mid + 1;\n        else hi = mid - 1;\n    }\n    return -1;\n}\n\nint main() {\n    std::cout << \"Found 7 at index \" << binarySearch({1, 3, 5, 7, 9, 11}, 7) << '\\n';\n    return 0;\n}\n",
            "java": "public class BinarySearch {\n    static int binarySearch(int[] a, int target) {\n        int lo = 0, hi = a.length - 1;\n        while (lo <= hi) {\n            int mid = lo + (hi - lo) / 2;\n            if (a[mid] == target) return mid;\n            if (a[mid] < target) lo = mid + 1;\n            else hi = mid - 1;\n        }\n        return -1;\n    }\n\n    public static void main(String[] args) {\n        System.out.println(\"Found 7 at index \" + binarySearch(new int[]{1, 3, 5, 7, 9, 11}, 7));\n    }\n}\n",
            "go": "package main\n\nimport \"fmt\"\n\nfunc binarySearch(a []int, target int) int {\n\tlo, hi := 0, len(a)-1\n\tfor lo <= hi {\n\t\tmid := lo + (hi-lo)/2\n\t\tswitch {\n\t\tcase a[mid] == target:\n\t\t\treturn mid\n\t\tcase a[mid] < target:\n\t\t\tlo = mid + 1\n\t\tdefault:\n\t\t\thi = mid - 1\n\t\t}\n\t}\n\treturn -1\n}\n\nfunc main() {\n\tfmt.Printf(\"Found 7 at index %d\\n\", binarySearch([]int{1, 3, 5, 7, 9, 11}, 7))\n}\n",
            "rust": "fn binary_search(a: &[i32], target: i32) -> i32 {\n    let (mut lo, mut hi) = (0i32, a.len() as i32 - 1);\n    while lo <= hi {\n        let mid = lo + (hi - lo) / 2;\n        let v = a[mid as usize];\n        if v == target {\n            return mid;\n        } else if v < target {\n            lo = mid + 1;\n        } else {\n            hi = mid - 1;\n        }\n    }\n    -1\n}\n\nfn main() {\n    println!(\"Found 7 at index {}\", binary_search(&[1, 3, 5, 7, 9, 11], 7));\n}\n",
            "ruby": "def binary_search(items, target)\n  lo = 0\n  hi = items.length - 1\n  while lo <= hi\n    mid = (lo + hi) / 2\n    return mid if items[mid] == target\n    if items[mid] < target\n      lo = mid + 1\n    else\n      hi = mid - 1\n    end\n  end\n  -1\nend\n\nputs \"Found 7 at index #{binary_search([1, 3, 5, 7, 9, 11], 7)}\"\n",
            "php": "<?php\nfunction binarySearch(array $a, int $target): int {\n    $lo = 0;\n    $hi = count($a) - 1;\n    while ($lo <= $hi) {\n        $mid = intdiv($lo + $hi, 2);\n        if ($a[$mid] == $target) return $mid;\n        if ($a[$mid] < $target) $lo = $mid + 1;\n        else $hi = $mid - 1;\n    }\n    return -1;\n}\n\necho \"Found 7 at index \" . binarySearch([1, 3, 5, 7, 9, 11], 7) . \"\\n\";\n",
        },
    },
    "gcd": {
        "title": "Greatest common divisor", "cls": "Gcd",
        "asks": ["Write a GCD program in {L}", "gcd of two numbers in {l}", "{L} function to find the greatest common divisor", "Euclid's algorithm in {L}"],
        "desc": "finds the greatest common divisor with Euclid's algorithm",
        "expected": "GCD of 48 and 18 is 6\n",
        "impl": {
            "python": "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n\n\nprint(f\"GCD of 48 and 18 is {gcd(48, 18)}\")\n",
            "javascript": "function gcd(a, b) {\n  while (b) [a, b] = [b, a % b];\n  return a;\n}\n\nconsole.log(`GCD of 48 and 18 is ${gcd(48, 18)}`);\n",
            "typescript": "function gcd(a: number, b: number): number {\n  while (b) [a, b] = [b, a % b];\n  return a;\n}\n\nconsole.log(`GCD of 48 and 18 is ${gcd(48, 18)}`);\n",
            "c": "#include <stdio.h>\n\nint gcd(int a, int b) {\n    while (b) {\n        int t = a % b;\n        a = b;\n        b = t;\n    }\n    return a;\n}\n\nint main(void) {\n    printf(\"GCD of 48 and 18 is %d\\n\", gcd(48, 18));\n    return 0;\n}\n",
            "cpp": "#include <iostream>\n\nint gcd(int a, int b) {\n    while (b) {\n        int t = a % b;\n        a = b;\n        b = t;\n    }\n    return a;\n}\n\nint main() {\n    std::cout << \"GCD of 48 and 18 is \" << gcd(48, 18) << '\\n';\n    return 0;\n}\n",
            "java": "public class Gcd {\n    static int gcd(int a, int b) {\n        while (b != 0) {\n            int t = a % b;\n            a = b;\n            b = t;\n        }\n        return a;\n    }\n\n    public static void main(String[] args) {\n        System.out.println(\"GCD of 48 and 18 is \" + gcd(48, 18));\n    }\n}\n",
            "go": "package main\n\nimport \"fmt\"\n\nfunc gcd(a, b int) int {\n\tfor b != 0 {\n\t\ta, b = b, a%b\n\t}\n\treturn a\n}\n\nfunc main() {\n\tfmt.Printf(\"GCD of 48 and 18 is %d\\n\", gcd(48, 18))\n}\n",
            "rust": "fn gcd(mut a: u64, mut b: u64) -> u64 {\n    while b != 0 {\n        let t = a % b;\n        a = b;\n        b = t;\n    }\n    a\n}\n\nfn main() {\n    println!(\"GCD of 48 and 18 is {}\", gcd(48, 18));\n}\n",
            "ruby": "def gcd(a, b)\n  a, b = b, a % b while b != 0\n  a\nend\n\nputs \"GCD of 48 and 18 is #{gcd(48, 18)}\"\n",
            "php": "<?php\nfunction gcd(int $a, int $b): int {\n    while ($b != 0) {\n        [$a, $b] = [$b, $a % $b];\n    }\n    return $a;\n}\n\necho \"GCD of 48 and 18 is \" . gcd(48, 18) . \"\\n\";\n",
        },
    },
}

FILE_BASE = {"fizzbuzz": "fizzbuzz", "factorial": "factorial", "fibonacci": "fibonacci", "primes": "primes", "reverse": "reverse_string",
             "bubble_sort": "bubble_sort", "binary_search": "binary_search", "gcd": "gcd"}


def filename(algo, lang):
    ext = LANGS[lang][1]
    return f"{ALGOS[algo]['cls']}.java" if lang == "java" else f"{FILE_BASE[algo]}.{ext}"


def held_out(algo, lang):
    """About 1 in 7 (algorithm, language) pairs are reserved for the test set."""
    return (sum(map(ord, algo + lang)) % 7) == 0
