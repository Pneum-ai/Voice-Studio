/** Speech and transcription model lifecycle management with live job progress. */

import {
  Box,
  CheckCircle2,
  CircleSlash2,
  CloudDownload,
  Cpu,
  HardDrive,
  Languages,
  LoaderCircle,
  MemoryStick,
  RefreshCw,
  Radio,
  Trash2,
} from "lucide-react";
import { useState } from "react";

import { api } from "../api/client";
import { Modal } from "../components/Modal";
import { useApp } from "../state/AppContext";
import { useToast } from "../state/ToastContext";
import { formatBytes, formatDate, problemMessage } from "../utils/format";

// Approximate complete repository downloads, checked 2026-10-26; decimal GB.
// MLX and NVIDIA repositories have equivalent sizes, including speech tokenizers.
const SPEECH_DOWNLOAD_GB = {
  "qwen-tts-0.6B": 2.52,
  "qwen-tts-1.7B": 4.54,
  "qwen-tts-1.7B-customvoice": 4.52,
};

const LANGUAGE_NAMES = new Intl.DisplayNames(["en"], { type: "language" });

/** Turn a catalog language code into a stable, readable English label. */
function languageLabel(code) {
  const name = LANGUAGE_NAMES.of(code) || code;
  return `${name} (${code})`;
}

/** Describe whether a transcription language is detected, hinted, or required. */
function languageModeLabel(mode) {
  return {
    automatic: "Detected automatically",
    optional: "Automatic with optional hint",
    required: "Must be selected",
  }[mode] || mode;
}

/** Render one model state card and its valid lifecycle actions. */
function ModelCard({ model, jobs, onAction, onDetail }) {
  const activeJob = jobs.find((job) => (
    job.id === model.active_job_id || (
      job.resource_id === model.id
      && ["queued", "running"].includes(job.state)
    )
  ));
  const loaded = model.load_state === "loaded";
  const downloaded = model.download_state === "downloaded";
  const streaming = model.kind === "tts"
    ? model.native_streaming
    : model.transcription?.streaming;
  const languageCount = model.transcription?.languages?.length || 0;

  return (
    <article className={`panel model-card${loaded ? " is-loaded" : ""}`}>
      <header>
        <span className="model-icon">
          {model.kind === "tts" ? <Radio size={20} /> : <MemoryStick size={20} />}
        </span>
        <div>
          <h3>{model.display_name}</h3>
          <p>
            {model.kind === "tts"
              ? "Speech generation"
              : `${model.provider} · ${model.family}`}
          </p>
        </div>
        <span className={`model-state${loaded ? " is-ready" : ""}`}>
          {loaded ? "Loaded" : downloaded ? "Downloaded" : "Absent"}
        </span>
      </header>
      <div className="model-facts">
        <span>
          <HardDrive size={14} />
          {model.size_bytes == null
            ? "Not downloaded"
            : `${formatBytes(model.size_bytes)} cached`}
        </span>
        {!downloaded && SPEECH_DOWNLOAD_GB[model.id] && (
          <span>About {SPEECH_DOWNLOAD_GB[model.id]} GB download</span>
        )}
        {model.kind === "transcription" && (
          <span><Languages size={14} />{languageCount} languages</span>
        )}
        <span>
          {streaming ? <Radio size={14} /> : <Cpu size={14} />}
          {model.kind === "tts"
            ? streaming ? "Native streaming" : "Complete waveform"
            : streaming ? "Streaming capable" : "File transcription"}
        </span>
      </div>
      {!model.backend_supported && (
        <p className="unsupported-note">
          <CircleSlash2 size={15} />
          Optional runtime is not installed or is incompatible with this platform.
        </p>
      )}
      {activeJob && (
        <div className="job-progress">
          <div>
            <span>{activeJob.stage}</span>
            <strong>{Math.round(activeJob.progress * 100)}%</strong>
          </div>
          <progress
            aria-label={`${model.display_name} progress`}
            max="1"
            value={activeJob.progress}
          />
          {activeJob.detail && <p>{activeJob.detail}</p>}
        </div>
      )}
      <footer>
        <button className="text-button" onClick={() => onDetail(model.id)} type="button">
          Details
        </button>
        <div className="model-actions">
          {!loaded && (
            <button
              className="button primary small"
              disabled={!model.backend_supported || Boolean(activeJob)}
              onClick={() => onAction(model.id, "loaded")}
              type="button"
            >
              {activeJob
                ? <LoaderCircle className="spin" size={15} />
                : <CloudDownload size={15} />}
              Load
            </button>
          )}
          {loaded && (
            <button
              className="button outline small"
              disabled={Boolean(activeJob)}
              onClick={() => onAction(model.id, "downloaded")}
              type="button"
            >
              Unload
            </button>
          )}
          {downloaded && !loaded && (
            <button
              aria-label={`Remove ${model.display_name}`}
              className="icon-button danger-text"
              disabled={Boolean(activeJob)}
              onClick={() => onAction(model.id, "absent")}
              type="button"
            >
              <Trash2 size={16} />
            </button>
          )}
        </div>
      </footer>
    </article>
  );
}

/** Render grouped model catalogs, details, actions, and retained jobs. */
export function ModelsView() {
  const {
    jobs,
    models,
    refreshJobs,
    refreshModels,
    startModelTransition,
  } = useApp();
  const { showToast } = useToast();
  const [detail, setDetail] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [jobDetail, setJobDetail] = useState(null);

  /** Fetch current model details for the inspection dialog. */
  const openModel = async (modelId) => {
    setDetailLoading(true);
    setDetail({ id: modelId });
    try {
      setDetail(await api.getModel(modelId));
    } catch (error) {
      setDetail(null);
      showToast(problemMessage(error), "error");
    } finally {
      setDetailLoading(false);
    }
  };

  /** Fetch current job details for the inspection dialog. */
  const openJob = async (jobId) => {
    try {
      setJobDetail(await api.getJob(jobId));
    } catch (error) {
      showToast(problemMessage(error), "error");
    }
  };

  /** Refresh models and jobs while preserving shared selections. */
  const refresh = async () => {
    try {
      await Promise.all([refreshModels(), refreshJobs()]);
      showToast("Model and job states refreshed.", "success");
    } catch (error) {
      showToast(problemMessage(error), "error");
    }
  };

  const ttsModels = models.filter((model) => model.kind === "tts");
  const sttModels = models.filter((model) => model.kind === "transcription");

  return (
    <div className="models-page">
      <section className="page-title-row">
        <div>
          <h1>Models</h1>
          <p>Download, load, unload, and remove local inference models.</p>
          <div className="page-title-actions">
            <button className="button outline" onClick={refresh} type="button">
              <RefreshCw size={16} />
              Refresh
            </button>
          </div>
        </div>
      </section>

      <section className="model-section">
        <div className="subsection-heading">
          <div>
            <h2>Qwen speech models</h2>
            <p>
              One speech model remains loaded at a time. Downloads need additional disk space
              for runtime packages; download size does not indicate memory use. First load can take
              several minutes.
            </p>
          </div>
          <span>{ttsModels.length} available</span>
        </div>
        <div className="model-grid">
          {ttsModels.map((model) => (
            <ModelCard
              jobs={jobs}
              key={model.id}
              model={model}
              onAction={startModelTransition}
              onDetail={openModel}
            />
          ))}
        </div>
      </section>

      <section className="model-section">
        <div className="subsection-heading">
          <div>
            <h2>Transcription models</h2>
            <p>Choose a recognition family that fits your language and hardware.</p>
          </div>
          <span>{sttModels.length} available</span>
        </div>
        <div className="model-grid">
          {sttModels.map((model) => (
            <ModelCard
              jobs={jobs}
              key={model.id}
              model={model}
              onAction={startModelTransition}
              onDetail={openModel}
            />
          ))}
        </div>
      </section>

      <section className="panel jobs-panel">
        <div className="subsection-heading">
          <div>
            <h2>Recent model jobs</h2>
            <p>Live progress is delivered by the shared event stream.</p>
          </div>
          <span>{jobs.length} retained</span>
        </div>
        {jobs.length ? (
          <div className="job-list">
            {jobs.map((job) => (
              <button key={job.id} onClick={() => openJob(job.id)} type="button">
                <span className={`job-state job-${job.state}`}>
                  {job.state === "complete"
                    ? <CheckCircle2 size={16} />
                    : <LoaderCircle
                      className={job.state === "running" ? "spin" : ""}
                      size={16}
                    />}
                </span>
                <span>
                  <strong>{job.resource_id}</strong>
                  <small>{job.stage}</small>
                  {job.detail && <small>{job.detail}</small>}
                </span>
                <progress max="1" value={job.progress} />
                <time>{formatDate(job.updated_at)}</time>
              </button>
            ))}
          </div>
        ) : (
          <div className="inline-empty">
            <Box size={20} />
            No model jobs have run yet.
          </div>
        )}
      </section>

      <Modal
        onClose={() => setDetail(null)}
        open={Boolean(detail)}
        title="Model details"
      >
        {detailLoading || !detail?.display_name ? (
          <p className="modal-loading">Loading model…</p>
        ) : (
          <div className="model-detail">
            <span className="model-icon large"><Cpu size={24} /></span>
            <h3>{detail.display_name}</h3>
            <p>{detail.id}</p>
            <dl>
              <div><dt>Kind</dt><dd>{detail.kind}</dd></div>
              <div><dt>Family</dt><dd>{detail.family}</dd></div>
              <div><dt>Provider</dt><dd>{detail.provider}</dd></div>
              <div><dt>Download</dt><dd>{detail.download_state}</dd></div>
              <div><dt>Memory</dt><dd>{detail.load_state}</dd></div>
              <div><dt>Size</dt><dd>{formatBytes(detail.size_bytes)}</dd></div>
              <div>
                <dt>Backend</dt>
                <dd>{detail.backend_supported ? "Supported" : "Unsupported"}</dd>
              </div>
              <div>
                <dt>Streaming</dt>
                <dd>
                  {(detail.kind === "transcription"
                    ? detail.transcription?.streaming
                    : detail.native_streaming)
                    ? "Supported"
                    : "Not supported"}
                </dd>
              </div>
              {detail.transcription && (
                <div>
                  <dt>Language selection</dt>
                  <dd>{languageModeLabel(detail.transcription.language_mode)}</dd>
                </div>
              )}
            </dl>
            {detail.transcription && (
              <section className="model-languages">
                <header>
                  <h4>Supported languages</h4>
                  <span>{detail.transcription.languages.length} total</span>
                </header>
                <ul>
                  {[...detail.transcription.languages]
                    .sort((left, right) => (
                      languageLabel(left).localeCompare(languageLabel(right))
                    ))
                    .map((code) => <li key={code}>{languageLabel(code)}</li>)}
                </ul>
              </section>
            )}
          </div>
        )}
      </Modal>

      <Modal
        onClose={() => setJobDetail(null)}
        open={Boolean(jobDetail)}
        title="Job details"
      >
        {jobDetail && (
          <div className="job-detail">
            <span className={`job-state job-${jobDetail.state}`}>
              {jobDetail.state}
            </span>
            <h3>{jobDetail.resource_id}</h3>
            <p>{jobDetail.stage}</p>
            <progress max="1" value={jobDetail.progress} />
            {jobDetail.detail && <p>{jobDetail.detail}</p>}
            <dl>
              <div><dt>Kind</dt><dd>{jobDetail.kind}</dd></div>
              <div><dt>Created</dt><dd>{formatDate(jobDetail.created_at)}</dd></div>
              <div><dt>Updated</dt><dd>{formatDate(jobDetail.updated_at)}</dd></div>
              {jobDetail.error && <div><dt>Error</dt><dd>{jobDetail.error}</dd></div>}
            </dl>
          </div>
        )}
      </Modal>
    </div>
  );
}
