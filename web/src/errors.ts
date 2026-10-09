/** An operation the backend refused, with the HTTP status the server would have answered. */
export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly detail: string,
  ) {
    super(detail);
  }
}
