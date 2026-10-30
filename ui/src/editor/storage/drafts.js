/** IndexedDB persistence for one browser-local editor draft and its audio assets. */

const DATABASE_NAME = "pneum-voice-editor";
const DATABASE_VERSION = 1;
const DRAFT_STORE = "drafts";
const ASSET_STORE = "assets";
const ACTIVE_DRAFT = "active";

/** Open the editor database and create stores on first use. */
function openDatabase() {
  return new Promise((resolve, reject) => {
    if (!globalThis.indexedDB) {
      reject(new Error("This browser does not support local draft storage."));
      return;
    }
    const request = indexedDB.open(DATABASE_NAME, DATABASE_VERSION);
    request.onupgradeneeded = () => {
      const database = request.result;
      if (!database.objectStoreNames.contains(DRAFT_STORE)) {
        database.createObjectStore(DRAFT_STORE);
      }
      if (!database.objectStoreNames.contains(ASSET_STORE)) {
        database.createObjectStore(ASSET_STORE);
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

/** Complete one IndexedDB request and close its database. */
async function runRequest(storeName, mode, operation) {
  const database = await openDatabase();
  return new Promise((resolve, reject) => {
    const transaction = database.transaction(storeName, mode);
    const request = operation(transaction.objectStore(storeName));
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
    transaction.oncomplete = () => database.close();
    transaction.onerror = () => reject(transaction.error);
  });
}

/** Load the current local project draft. */
export function loadDraft() {
  return runRequest(DRAFT_STORE, "readonly", (store) => store.get(ACTIVE_DRAFT));
}

/** Save a serializable project snapshot. */
export function saveDraft(project) {
  return runRequest(
    DRAFT_STORE,
    "readwrite",
    (store) => store.put(project, ACTIVE_DRAFT),
  );
}

/** Store an uploaded or recorded audio blob by its source identifier. */
export function putAsset(assetId, blob) {
  return runRequest(ASSET_STORE, "readwrite", (store) => store.put(blob, assetId));
}

/** Load one uploaded or recorded audio blob. */
export function getAsset(assetId) {
  return runRequest(ASSET_STORE, "readonly", (store) => store.get(assetId));
}

/** Remove the active edit decision list while retaining reusable local assets. */
export function clearDraft() {
  return runRequest(DRAFT_STORE, "readwrite", (store) => store.delete(ACTIVE_DRAFT));
}
