defmodule ChaoticEchoes.Sorting.Algorithms do
  @moduledoc """
  Implementação de algoritmos de ordenação com rastreamento de estados para visualização.
  """

  @doc """
  Retorna lista de todos os algoritmos disponíveis.
  """
  def list_algorithms do
    [
      %{
        id: "bubble",
        name: "Bubble Sort",
        complexity: "O(n²)",
        code: """
def bubble_sort(array):
    n = len(array)
    for i in range(n):
        for j in range(0, n - i - 1):
            if array[j] > array[j + 1]:
                array[j], array[j + 1] = array[j + 1], array[j]
    return array
"""
      },
      %{
        id: "quick",
        name: "Quick Sort",
        complexity: "O(n log n)",
        code: """
def quick_sort(array, low, high):
    if low < high:
        pivot = partition(array, low, high)
        quick_sort(array, low, pivot - 1)
        quick_sort(array, pivot + 1, high)
    return array

def partition(array, low, high):
    pivot = array[high]
    i = low - 1
    for j in range(low, high):
        if array[j] <= pivot:
            i += 1
            array[i], array[j] = array[j], array[i]
    array[i + 1], array[high] = array[high], array[i + 1]
    return i + 1
"""
      },
      %{
        id: "merge",
        name: "Merge Sort",
        complexity: "O(n log n)",
        code: """
def merge_sort(array):
    if len(array) <= 1:
        return array

    mid = len(array) // 2
    left = merge_sort(array[:mid])
    right = merge_sort(array[mid:])

    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])
    return result
"""
      },
      %{
        id: "heap",
        name: "Heap Sort",
        complexity: "O(n log n)",
        code: """
def heap_sort(array):
    n = len(array)

    # Build max heap
    for i in range(n // 2 - 1, -1, -1):
        heapify(array, n, i)

    # Extract elements from heap
    for i in range(n - 1, 0, -1):
        array[0], array[i] = array[i], array[0]
        heapify(array, i, 0)

    return array

def heapify(array, n, i):
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2

    if left < n and array[left] > array[largest]:
        largest = left
    if right < n and array[right] > array[largest]:
        largest = right

    if largest != i:
        array[i], array[largest] = array[largest], array[i]
        heapify(array, n, largest)
"""
      },
      %{
        id: "insertion",
        name: "Insertion Sort",
        complexity: "O(n²)",
        code: """
def insertion_sort(array):
    for i in range(1, len(array)):
        key = array[i]
        j = i - 1

        while j >= 0 and array[j] > key:
            array[j + 1] = array[j]
            j -= 1

        array[j + 1] = key

    return array
"""
      },
      %{
        id: "selection",
        name: "Selection Sort",
        complexity: "O(n²)",
        code: """
def selection_sort(array):
    n = len(array)

    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if array[j] < array[min_idx]:
                min_idx = j

        array[i], array[min_idx] = array[min_idx], array[i]

    return array
"""
      },
      %{
        id: "shell",
        name: "Shell Sort",
        complexity: "O(n log n)",
        code: """
def shell_sort(array):
    n = len(array)
    gap = n // 2

    while gap > 0:
        for i in range(gap, n):
            temp = array[i]
            j = i

            while j >= gap and array[j - gap] > temp:
                array[j] = array[j - gap]
                j -= gap

            array[j] = temp

        gap //= 2

    return array
"""
      },
      %{
        id: "counting",
        name: "Counting Sort",
        complexity: "O(n + k)",
        code: """
def counting_sort(array):
    if not array:
        return array

    max_val = max(array)
    min_val = min(array)
    range_size = max_val - min_val + 1

    count = [0] * range_size
    output = [0] * len(array)

    # Count occurrences
    for num in array:
        count[num - min_val] += 1

    # Accumulate counts
    for i in range(1, len(count)):
        count[i] += count[i - 1]

    # Build output array
    for num in reversed(array):
        output[count[num - min_val] - 1] = num
        count[num - min_val] -= 1

    return output
"""
      },
      %{
        id: "radix",
        name: "Radix Sort",
        complexity: "O(d × n)",
        code: """
def radix_sort(array):
    if not array:
        return array

    max_val = max(array)
    exp = 1

    while max_val // exp > 0:
        counting_sort_by_digit(array, exp)
        exp *= 10

    return array

def counting_sort_by_digit(array, exp):
    n = len(array)
    output = [0] * n
    count = [0] * 10

    for num in array:
        digit = (num // exp) % 10
        count[digit] += 1

    for i in range(1, 10):
        count[i] += count[i - 1]

    for num in reversed(array):
        digit = (num // exp) % 10
        output[count[digit] - 1] = num
        count[digit] -= 1

    for i in range(n):
        array[i] = output[i]
"""
      },
      %{
        id: "bucket",
        name: "Bucket Sort",
        complexity: "O(n + k)",
        code: """
def bucket_sort(array):
    if not array:
        return array

    # Create buckets
    bucket_count = len(array)
    max_val = max(array)
    min_val = min(array)
    range_size = max_val - min_val + 1

    buckets = [[] for _ in range(bucket_count)]

    # Distribute elements into buckets
    for num in array:
        idx = min(bucket_count - 1,
                  (num - min_val) * bucket_count // range_size)
        buckets[idx].append(num)

    # Sort each bucket and concatenate
    result = []
    for bucket in buckets:
        result.extend(sorted(bucket))

    return result
"""
      }
    ]
  end

  @doc """
  Executa o algoritmo selecionado e retorna snapshots dos estados.
  """
  def sort(algorithm, array) do
    case algorithm do
      "bubble" -> bubble_sort(array)
      "quick" -> quick_sort(array)
      "merge" -> merge_sort(array)
      "heap" -> heap_sort(array)
      "insertion" -> insertion_sort(array)
      "selection" -> selection_sort(array)
      "shell" -> shell_sort(array)
      "counting" -> counting_sort(array)
      "radix" -> radix_sort(array)
      "bucket" -> bucket_sort(array)
      _ -> {:error, "Algoritmo não encontrado"}
    end
  end

  # Bubble Sort
  defp bubble_sort(array) do
    snapshots = []
    n = length(array)
    {sorted, all_snapshots} = bubble_sort_pass(array, n, 0, snapshots)
    {:ok, all_snapshots ++ [%{array: sorted, comparing: [], swapping: [], sorted: Enum.to_list(0..(n-1))}]}
  end

  defp bubble_sort_pass(array, n, pass, snapshots) when pass >= n - 1 do
    {array, snapshots}
  end

  defp bubble_sort_pass(array, n, pass, snapshots) do
    {new_array, new_snapshots} = bubble_sort_iteration(array, 0, n - pass - 1, snapshots, pass)
    bubble_sort_pass(new_array, n, pass + 1, new_snapshots)
  end

  defp bubble_sort_iteration(array, i, max, snapshots, _pass) when i >= max do
    {array, snapshots}
  end

  defp bubble_sort_iteration(array, i, max, snapshots, pass) do
    val1 = Enum.at(array, i)
    val2 = Enum.at(array, i + 1)

    snapshot = %{
      array: array,
      comparing: [i, i + 1],
      swapping: [],
      sorted: if(pass > 0, do: Enum.to_list((length(array) - pass)..(length(array) - 1)), else: [])
    }

    if val1 > val2 do
      new_array = swap(array, i, i + 1)
      swap_snapshot = %{
        array: new_array,
        comparing: [],
        swapping: [i, i + 1],
        sorted: if(pass > 0, do: Enum.to_list((length(array) - pass)..(length(array) - 1)), else: [])
      }
      bubble_sort_iteration(new_array, i + 1, max, snapshots ++ [snapshot, swap_snapshot], pass)
    else
      bubble_sort_iteration(array, i + 1, max, snapshots ++ [snapshot], pass)
    end
  end

  # Quick Sort
  defp quick_sort(array) do
    snapshots = []
    {sorted, all_snapshots} = quick_sort_helper(array, 0, length(array) - 1, snapshots)
    {:ok, all_snapshots ++ [%{array: sorted, comparing: [], swapping: [], sorted: Enum.to_list(0..(length(array)-1))}]}
  end

  defp quick_sort_helper(array, low, high, snapshots) when low < high do
    {array, pivot_index, new_snapshots} = partition(array, low, high, snapshots)
    {array, snapshots1} = quick_sort_helper(array, low, pivot_index - 1, new_snapshots)
    quick_sort_helper(array, pivot_index + 1, high, snapshots1)
  end

  defp quick_sort_helper(array, _low, _high, snapshots), do: {array, snapshots}

  defp partition(array, low, high, snapshots) do
    pivot = Enum.at(array, high)
    i = low - 1

    {array, i, snapshots} =
      Enum.reduce(low..(high - 1), {array, i, snapshots}, fn j, {arr, idx, snaps} ->
        snapshot = %{
          array: arr,
          comparing: [j, high],
          swapping: [],
          sorted: [],
          partitions: [{low, high, :active}]
        }

        if Enum.at(arr, j) <= pivot do
          new_idx = idx + 1
          new_arr = swap(arr, new_idx, j)
          swap_snap = %{
            array: new_arr,
            comparing: [],
            swapping: [new_idx, j],
            sorted: [],
            partitions: [{low, high, :active}]
          }
          {new_arr, new_idx, snaps ++ [snapshot, swap_snap]}
        else
          {arr, idx, snaps ++ [snapshot]}
        end
      end)

    final_array = swap(array, i + 1, high)
    final_snapshot = %{
      array: final_array,
      comparing: [],
      swapping: [i + 1, high],
      sorted: [],
      partitions: [{low, high, :pivot}]
    }
    {final_array, i + 1, snapshots ++ [final_snapshot]}
  end

  # Merge Sort
  defp merge_sort(array) do
    snapshots = []
    {sorted, all_snapshots} = merge_sort_helper(array, snapshots)
    {:ok, all_snapshots ++ [%{array: sorted, comparing: [], swapping: [], sorted: Enum.to_list(0..(length(array)-1))}]}
  end

  defp merge_sort_helper(array, snapshots) when length(array) <= 1 do
    {array, snapshots}
  end

  defp merge_sort_helper(array, snapshots) do
    mid = div(length(array), 2)
    {left, right} = Enum.split(array, mid)

    {sorted_left, snapshots1} = merge_sort_helper(left, snapshots)
    {sorted_right, snapshots2} = merge_sort_helper(right, snapshots1)

    merge(sorted_left, sorted_right, snapshots2)
  end

  defp merge(left, right, snapshots) do
    merge_helper(left, right, [], snapshots)
  end

  defp merge_helper([], right, acc, snapshots) do
    result = acc ++ right
    snapshot = %{array: result, comparing: [], swapping: [], sorted: []}
    {result, snapshots ++ [snapshot]}
  end

  defp merge_helper(left, [], acc, snapshots) do
    result = acc ++ left
    snapshot = %{array: result, comparing: [], swapping: [], sorted: []}
    {result, snapshots ++ [snapshot]}
  end

  defp merge_helper([l | left_rest], [r | right_rest], acc, snapshots) do
    snapshot = %{array: acc ++ [l] ++ left_rest ++ [r] ++ right_rest, comparing: [length(acc), length(acc) + length(left_rest) + 1], swapping: [], sorted: []}

    if l <= r do
      merge_helper(left_rest, [r | right_rest], acc ++ [l], snapshots ++ [snapshot])
    else
      merge_helper([l | left_rest], right_rest, acc ++ [r], snapshots ++ [snapshot])
    end
  end

  # Insertion Sort
  defp insertion_sort(array) do
    {sorted, snapshots} = insertion_sort_helper(array, 1, [])
    {:ok, snapshots ++ [%{array: sorted, comparing: [], swapping: [], sorted: Enum.to_list(0..(length(array)-1))}]}
  end

  defp insertion_sort_helper(array, i, snapshots) when i >= length(array) do
    {array, snapshots}
  end

  defp insertion_sort_helper(array, i, snapshots) do
    key = Enum.at(array, i)
    {new_array, new_snapshots} = insert_key(array, i, i - 1, key, snapshots)
    insertion_sort_helper(new_array, i + 1, new_snapshots)
  end

  defp insert_key(array, original_pos, j, key, snapshots) when j >= 0 do
    if Enum.at(array, j) > key do
      snapshot = %{array: array, comparing: [j, original_pos], swapping: [], sorted: Enum.to_list(0..(original_pos - 1))}
      new_array = List.replace_at(array, j + 1, Enum.at(array, j))
      swap_snapshot = %{array: new_array, comparing: [], swapping: [j, j + 1], sorted: Enum.to_list(0..(original_pos - 1))}
      insert_key(new_array, original_pos, j - 1, key, snapshots ++ [snapshot, swap_snapshot])
    else
      new_array = List.replace_at(array, j + 1, key)
      {new_array, snapshots}
    end
  end

  defp insert_key(array, _original_pos, j, key, snapshots) do
    new_array = List.replace_at(array, j + 1, key)
    {new_array, snapshots}
  end

  # Selection Sort
  defp selection_sort(array) do
    {sorted, snapshots} = selection_sort_helper(array, 0, [])
    {:ok, snapshots ++ [%{array: sorted, comparing: [], swapping: [], sorted: Enum.to_list(0..(length(array)-1))}]}
  end

  defp selection_sort_helper(array, i, snapshots) when i >= length(array) - 1 do
    {array, snapshots}
  end

  defp selection_sort_helper(array, i, snapshots) do
    {min_idx, new_snapshots} = find_min(array, i, i + 1, i, snapshots)

    if min_idx != i do
      new_array = swap(array, i, min_idx)
      swap_snapshot = %{array: new_array, comparing: [], swapping: [i, min_idx], sorted: Enum.to_list(0..(i))}
      selection_sort_helper(new_array, i + 1, new_snapshots ++ [swap_snapshot])
    else
      selection_sort_helper(array, i + 1, new_snapshots)
    end
  end

  defp find_min(array, _start, j, min_idx, snapshots) when j >= length(array) do
    {min_idx, snapshots}
  end

  defp find_min(array, start, j, min_idx, snapshots) do
    snapshot = %{array: array, comparing: [j, min_idx], swapping: [], sorted: if(start > 0, do: Enum.to_list(0..(start - 1)), else: [])}

    if Enum.at(array, j) < Enum.at(array, min_idx) do
      find_min(array, start, j + 1, j, snapshots ++ [snapshot])
    else
      find_min(array, start, j + 1, min_idx, snapshots ++ [snapshot])
    end
  end

  # Shell Sort
  defp shell_sort(array) do
    n = length(array)
    gap = div(n, 2)
    {sorted, snapshots} = shell_sort_helper(array, gap, [])
    {:ok, snapshots ++ [%{array: sorted, comparing: [], swapping: [], sorted: Enum.to_list(0..(n-1))}]}
  end

  defp shell_sort_helper(array, gap, snapshots) when gap > 0 do
    {new_array, new_snapshots} = shell_sort_pass(array, gap, gap, snapshots)
    shell_sort_helper(new_array, div(gap, 2), new_snapshots)
  end

  defp shell_sort_helper(array, _gap, snapshots), do: {array, snapshots}

  defp shell_sort_pass(array, gap, i, snapshots) when i < length(array) do
    temp = Enum.at(array, i)
    {new_array, new_snapshots} = shell_insert(array, gap, i, temp, snapshots)
    shell_sort_pass(new_array, gap, i + 1, new_snapshots)
  end

  defp shell_sort_pass(array, _gap, _i, snapshots), do: {array, snapshots}

  defp shell_insert(array, gap, j, temp, snapshots) when j >= gap do
    if Enum.at(array, j - gap) > temp do
      snapshot = %{array: array, comparing: [j, j - gap], swapping: [], sorted: []}
      new_array = List.replace_at(array, j, Enum.at(array, j - gap))
      swap_snapshot = %{array: new_array, comparing: [], swapping: [j, j - gap], sorted: []}
      shell_insert(new_array, gap, j - gap, temp, snapshots ++ [snapshot, swap_snapshot])
    else
      new_array = List.replace_at(array, j, temp)
      {new_array, snapshots}
    end
  end

  defp shell_insert(array, _gap, j, temp, snapshots) do
    new_array = List.replace_at(array, j, temp)
    {new_array, snapshots}
  end

  # Counting Sort
  defp counting_sort(array) when length(array) == 0 do
    {:ok, [%{array: [], comparing: [], swapping: [], sorted: []}]}
  end

  defp counting_sort(array) do
    max_val = Enum.max(array)
    min_val = Enum.min(array)
    range = max_val - min_val + 1

    count = List.duplicate(0, range)

    # Contar ocorrências
    count = Enum.reduce(array, count, fn val, acc ->
      idx = val - min_val
      List.replace_at(acc, idx, Enum.at(acc, idx) + 1)
    end)

    # Acumular contagens
    count = Enum.scan(count, &(&1 + &2))

    # Construir array ordenado
    output = List.duplicate(0, length(array))

    {sorted, _} = Enum.reduce(Enum.reverse(array), {output, count}, fn val, {out, cnt} ->
      idx = val - min_val
      pos = Enum.at(cnt, idx) - 1
      new_out = List.replace_at(out, pos, val)
      new_cnt = List.replace_at(cnt, idx, pos)
      {new_out, new_cnt}
    end)

    {:ok, [
      %{array: array, comparing: [], swapping: [], sorted: []},
      %{array: sorted, comparing: [], swapping: [], sorted: Enum.to_list(0..(length(array)-1))}
    ]}
  end

  # Radix Sort
  defp radix_sort(array) when length(array) == 0 do
    {:ok, [%{array: [], comparing: [], swapping: [], sorted: []}]}
  end

  defp radix_sort(array) do
    max_val = Enum.max(array)
    {sorted, snapshots} = radix_sort_helper(array, 1, max_val, [])
    {:ok, snapshots ++ [%{array: sorted, comparing: [], swapping: [], sorted: Enum.to_list(0..(length(array)-1))}]}
  end

  defp radix_sort_helper(array, exp, max_val, snapshots) when div(max_val, exp) > 0 do
    {sorted, new_snapshots} = counting_sort_by_digit(array, exp, snapshots)
    radix_sort_helper(sorted, exp * 10, max_val, new_snapshots)
  end

  defp radix_sort_helper(array, _exp, _max_val, snapshots), do: {array, snapshots}

  defp counting_sort_by_digit(array, exp, snapshots) do
    n = length(array)
    output = List.duplicate(0, n)
    count = List.duplicate(0, 10)

    # Contar dígitos
    count = Enum.reduce(array, count, fn val, acc ->
      digit = rem(div(val, exp), 10)
      List.replace_at(acc, digit, Enum.at(acc, digit) + 1)
    end)

    # Acumular
    count = Enum.scan(count, &(&1 + &2))

    # Construir output
    {sorted, _} = Enum.reduce(Enum.reverse(array), {output, count}, fn val, {out, cnt} ->
      digit = rem(div(val, exp), 10)
      pos = Enum.at(cnt, digit) - 1
      new_out = List.replace_at(out, pos, val)
      new_cnt = List.replace_at(cnt, digit, pos)
      {new_out, new_cnt}
    end)

    snapshot = %{array: sorted, comparing: [], swapping: [], sorted: []}
    {sorted, snapshots ++ [snapshot]}
  end

  # Bucket Sort
  defp bucket_sort(array) when length(array) == 0 do
    {:ok, [%{array: [], comparing: [], swapping: [], sorted: []}]}
  end

  defp bucket_sort(array) do
    n = length(array)
    max_val = Enum.max(array)
    min_val = Enum.min(array)
    bucket_count = max(1, n)
    range = max_val - min_val + 1

    # Criar buckets
    buckets = List.duplicate([], bucket_count)

    # Distribuir elementos nos buckets
    buckets = Enum.reduce(array, buckets, fn val, acc ->
      idx = min(bucket_count - 1, div((val - min_val) * bucket_count, range))
      List.update_at(acc, idx, fn bucket -> bucket ++ [val] end)
    end)

    # Ordenar cada bucket e concatenar
    sorted = buckets
    |> Enum.map(&Enum.sort/1)
    |> List.flatten()

    {:ok, [
      %{array: array, comparing: [], swapping: [], sorted: []},
      %{array: sorted, comparing: [], swapping: [], sorted: Enum.to_list(0..(length(array)-1))}
    ]}
  end

  # Heap Sort
  defp heap_sort(array) do
    n = length(array)

    # Construir heap
    {heapified, snapshots} = build_heap(array, n, [])

    # Extrair elementos do heap
    {sorted, all_snapshots} = extract_from_heap(heapified, n - 1, snapshots)

    {:ok, all_snapshots ++ [%{array: sorted, comparing: [], swapping: [], sorted: Enum.to_list(0..(n-1))}]}
  end

  defp build_heap(array, n, snapshots) do
    start_idx = div(n, 2) - 1
    build_heap_helper(array, n, start_idx, snapshots)
  end

  defp build_heap_helper(array, n, i, snapshots) when i >= 0 do
    {heapified, new_snapshots} = heapify(array, n, i, snapshots)
    build_heap_helper(heapified, n, i - 1, new_snapshots)
  end

  defp build_heap_helper(array, _n, _i, snapshots), do: {array, snapshots}

  defp extract_from_heap(array, i, snapshots) when i > 0 do
    # Trocar raiz com último elemento
    swapped = swap(array, 0, i)
    swap_snapshot = %{array: swapped, comparing: [], swapping: [0, i], sorted: Enum.to_list((i + 1)..(length(array) - 1))}

    # Heapify na raiz reduzida
    {heapified, new_snapshots} = heapify(swapped, i, 0, snapshots ++ [swap_snapshot])

    extract_from_heap(heapified, i - 1, new_snapshots)
  end

  defp extract_from_heap(array, _i, snapshots), do: {array, snapshots}

  defp heapify(array, n, i, snapshots) do
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2

    largest = if left < n and Enum.at(array, left) > Enum.at(array, largest), do: left, else: largest
    largest = if right < n and Enum.at(array, right) > Enum.at(array, largest), do: right, else: largest

    if largest != i do
      snapshot = %{array: array, comparing: [i, largest], swapping: [], sorted: []}
      swapped = swap(array, i, largest)
      swap_snapshot = %{array: swapped, comparing: [], swapping: [i, largest], sorted: []}
      heapify(swapped, n, largest, snapshots ++ [snapshot, swap_snapshot])
    else
      {array, snapshots}
    end
  end

  # Helper function to swap elements
  defp swap(array, i, j) do
    val_i = Enum.at(array, i)
    val_j = Enum.at(array, j)

    array
    |> List.replace_at(i, val_j)
    |> List.replace_at(j, val_i)
  end
end
