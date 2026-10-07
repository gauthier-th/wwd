import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { createBrowserRouter, RouterProvider } from "react-router";

import { ApiError } from "./api";
import { DevicePage } from "./pages/Device";
import { ErrorPage, Layout } from "./pages/Layout";
import { Home } from "./pages/Home";
import { TrainingPage } from "./pages/Training";
import { WakeWordPage } from "./pages/WakeWord";
import "./styles.css";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // a 4xx won't fix itself
      retry: (count, err) => !(err instanceof ApiError && err.status < 500) && count < 2,
      refetchOnWindowFocus: false,
    },
  },
});

const router = createBrowserRouter([
  {
    element: <Layout />,
    errorElement: <ErrorPage />,
    children: [
      { path: "/", element: <Home /> },
      { path: "/w/:wid", element: <WakeWordPage /> },
      { path: "/w/:wid/t/:tid", element: <TrainingPage /> },
      { path: "/w/:wid/t/:tid/device", element: <DevicePage /> },
    ],
  },
]);

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
    </QueryClientProvider>
  </StrictMode>,
);
