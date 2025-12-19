defmodule ChaoticEchoesWeb.SortingVisualizerLive do
  use ChaoticEchoesWeb, :live_view
  alias ChaoticEchoes.Sorting.Algorithms

  @impl true
  def mount(_params, _session, socket) do
    {:ok,
     socket
     |> assign(:array, generate_random_array(50))
     |> assign(:array_size, 50)
     |> assign(:algorithm, "bubble")
     |> assign(:algorithms, Algorithms.list_algorithms())
     |> assign(:is_sorting, false)
     |> assign(:is_paused, false)
     |> assign(:current_snapshot_index, 0)
     |> assign(:snapshots, [])
     |> assign(:speed, 50)
     |> assign(:comparisons, 0)
     |> assign(:swaps, 0)}
  end

  @impl true
  def handle_event("update_array_size", %{"value" => size_str}, socket) do
    size = String.to_integer(size_str)

    {:noreply,
     socket
     |> assign(:array, generate_random_array(size))
     |> assign(:array_size, size)
     |> assign(:is_sorting, false)
     |> assign(:snapshots, [])
     |> assign(:current_snapshot_index, 0)
     |> assign(:comparisons, 0)
     |> assign(:swaps, 0)}
  end

  @impl true
  def handle_event("select_algorithm", %{"algorithm" => algorithm}, socket) do
    {:noreply, assign(socket, :algorithm, algorithm)}
  end

  @impl true
  def handle_event("update_speed", %{"value" => speed_str}, socket) do
    {:noreply, assign(socket, :speed, String.to_integer(speed_str))}
  end

  @impl true
  def handle_event("start_sort", _, socket) do
    if socket.assigns.is_sorting do
      {:noreply, socket}
    else
      case Algorithms.sort(socket.assigns.algorithm, socket.assigns.array) do
        {:ok, snapshots} ->
          send(self(), :animate_next_step)

          {:noreply,
           socket
           |> assign(:is_sorting, true)
           |> assign(:is_paused, false)
           |> assign(:snapshots, snapshots)
           |> assign(:current_snapshot_index, 0)
           |> assign(:comparisons, 0)
           |> assign(:swaps, 0)}

        {:error, _} ->
          {:noreply, socket}
      end
    end
  end

  @impl true
  def handle_event("pause_sort", _, socket) do
    {:noreply, assign(socket, :is_paused, !socket.assigns.is_paused)}
  end

  @impl true
  def handle_event("reset", _, socket) do
    {:noreply,
     socket
     |> assign(:array, generate_random_array(socket.assigns.array_size))
     |> assign(:is_sorting, false)
     |> assign(:is_paused, false)
     |> assign(:snapshots, [])
     |> assign(:current_snapshot_index, 0)
     |> assign(:comparisons, 0)
     |> assign(:swaps, 0)}
  end

  @impl true
  def handle_info(:animate_next_step, socket) do
    if socket.assigns.is_paused do
      Process.send_after(self(), :animate_next_step, 100)
      {:noreply, socket}
    else
      index = socket.assigns.current_snapshot_index

      if index < length(socket.assigns.snapshots) do
        snapshot = Enum.at(socket.assigns.snapshots, index)

        # Contar comparações e trocas
        comparisons = socket.assigns.comparisons + if length(snapshot.comparing) > 0, do: 1, else: 0
        swaps = socket.assigns.swaps + if length(snapshot.swapping) > 0, do: 1, else: 0

        # Agendar próximo passo
        delay = calculate_delay(socket.assigns.speed)
        Process.send_after(self(), :animate_next_step, delay)

        {:noreply,
         socket
         |> assign(:array, snapshot.array)
         |> assign(:current_snapshot_index, index + 1)
         |> assign(:comparisons, comparisons)
         |> assign(:swaps, swaps)
         |> push_event("update_visualization", %{
           array: snapshot.array,
           comparing: snapshot.comparing,
           swapping: snapshot.swapping,
           sorted: snapshot.sorted,
           partitions: Map.get(snapshot, :partitions, [])
         })}
      else
        {:noreply,
         socket
         |> assign(:is_sorting, false)
         |> assign(:is_paused, false)}
      end
    end
  end

  defp generate_random_array(size) do
    1..size
    |> Enum.map(fn _ -> Enum.random(10..500) end)
  end

  defp calculate_delay(speed) do
    # Speed vai de 1 (mais lento) a 100 (mais rápido)
    # Delay vai de 1000ms a 10ms
    max(10, 1010 - speed * 10)
  end

  defp get_algorithm_name(algorithms, id) do
    algorithm = Enum.find(algorithms, fn alg -> alg.id == id end)
    if algorithm, do: algorithm.name, else: "Unknown"
  end

  defp get_algorithm_complexity(algorithms, id) do
    algorithm = Enum.find(algorithms, fn alg -> alg.id == id end)
    if algorithm, do: algorithm.complexity, else: "N/A"
  end

  defp get_algorithm_code(algorithms, id) do
    algorithm = Enum.find(algorithms, fn alg -> alg.id == id end)
    if algorithm, do: algorithm.code, else: "# Código não disponível"
  end
end
