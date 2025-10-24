defmodule ChaoticEchoes.Repo do
  use Ecto.Repo,
    otp_app: :chaotic_echoes,
    adapter: Ecto.Adapters.Postgres
end
