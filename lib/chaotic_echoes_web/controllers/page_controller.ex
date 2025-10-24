defmodule ChaoticEchoesWeb.PageController do
  use ChaoticEchoesWeb, :controller

  def home(conn, _params) do
    render(conn, :home)
  end
end
