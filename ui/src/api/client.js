/** Same-origin REST client for the complete v1 voice API. */

const JSON_HEADERS = {
  Accept: "application/json",
  "Content-Type": "application/json",
};

/** Represent a structured API problem response. */
export class ApiError extends Error {
  /** Preserve the problem payload and originating response. */
  constructor(problem, response) {
    super(problem.detail || problem.title || `Request failed (${response.status})`);
    this.name = "ApiError";
    this.status = response.status;
    this.code = problem.code || "request_failed";
    this.problem = problem;
  }
}

/** Convert a failed response into the shared problem shape. */
async function parseProblem(response) {
  try {
    return await response.json();
  } catch {
    return {
      title: "Request failed",
      detail: `${response.status} ${response.statusText}`,
      status: response.status,
    };
  }
}

/** Send one request and reject non-successful problem responses. */
async function request(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    throw new ApiError(await parseProblem(response), response);
  }
  return response;
}

/** Send one request and decode its JSON response. */
async function jsonRequest(path, options = {}) {
  const response = await request(path, options);
  return response.json();
}

/** Build standard JSON fetch options. */
function jsonOptions(method, payload) {
  return {
    method,
    headers: JSON_HEADERS,
    body: payload === undefined ? undefined : JSON.stringify(payload),
  };
}

/** Serialize defined query values without empty parameters. */
function queryString(values) {
  const params = new URLSearchParams();
  Object.entries(values).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "") {
      params.set(key, String(value));
    }
  });
  const rendered = params.toString();
  return rendered ? `?${rendered}` : "";
}

/** Append optional browser trim coordinates to one multipart request. */
function appendTrim(body, trim) {
  if (!trim) {
    return;
  }
  body.append("trim_start_seconds", String(trim.startSeconds));
  body.append("trim_end_seconds", String(trim.endSeconds));
}

export const api = {
  health: () => jsonRequest("/health"),

  listProfiles: () => jsonRequest("/v1/profiles"),
  getProfile: (profileId) => jsonRequest(`/v1/profiles/${profileId}`),
  createProfile: (name) => (
    jsonRequest("/v1/profiles", jsonOptions("POST", { name }))
  ),
  renameProfile: (profileId, name) => (
    jsonRequest(`/v1/profiles/${profileId}`, jsonOptions("PATCH", { name }))
  ),
  deleteProfile: (profileId) => (
    jsonRequest(`/v1/profiles/${profileId}`, { method: "DELETE" })
  ),

  listSamples: (profileId) => (
    jsonRequest(`/v1/profiles/${profileId}/samples`)
  ),
  addSample: (profileId, file, transcript, trim = null) => {
    const body = new FormData();
    body.append("audio", file);
    body.append("transcript", transcript);
    appendTrim(body, trim);
    return jsonRequest(`/v1/profiles/${profileId}/samples`, {
      method: "POST",
      body,
    });
  },
  updateSample: (profileId, sampleId, transcript) => (
    jsonRequest(
      `/v1/profiles/${profileId}/samples/${sampleId}`,
      jsonOptions("PATCH", { transcript }),
    )
  ),
  deleteSample: (profileId, sampleId) => (
    jsonRequest(`/v1/profiles/${profileId}/samples/${sampleId}`, {
      method: "DELETE",
    })
  ),

  generateSpeech: async (payload) => {
    const response = await request("/v1/speech", jsonOptions("POST", payload));
    return response.blob();
  },

  listGenerations: (filters = {}) => (
    jsonRequest(`/v1/generations${queryString(filters)}`)
  ),
  createGeneration: (payload) => (
    jsonRequest("/v1/generations", jsonOptions("POST", payload))
  ),
  getGeneration: (generationId) => (
    jsonRequest(`/v1/generations/${generationId}`)
  ),
  deleteGeneration: (generationId) => (
    jsonRequest(`/v1/generations/${generationId}`, { method: "DELETE" })
  ),

  transcribe: (file, model, language, context, trim = null) => {
    const body = new FormData();
    body.append("audio", file);
    body.append("model", model);
    if (language) {
      body.append("language", language);
    }
    if (context) {
      body.append("context", context);
    }
    appendTrim(body, trim);
    return jsonRequest("/v1/transcriptions", { method: "POST", body });
  },

  listModels: () => jsonRequest("/v1/models"),
  getModel: (modelId) => jsonRequest(`/v1/models/${modelId}`),
  setModelState: (modelId, state) => (
    jsonRequest(`/v1/models/${modelId}`, jsonOptions("PATCH", { state }))
  ),

  listJobs: () => jsonRequest("/v1/jobs"),
  getJob: (jobId) => jsonRequest(`/v1/jobs/${jobId}`),
  clearProfilePrompts: () => (
    jsonRequest("/v1/cache/profile-prompts", { method: "DELETE" })
  ),
};
