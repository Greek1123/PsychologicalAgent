export type ViewRole = "student" | "counselor" | "research" | "admin";
export type HumanInterventionStatus = "acknowledged" | "in_progress" | "escalated" | "resolved" | "closed";

export interface ConversationTurn {
  role: "user" | "assistant" | "system" | string;
  content: string;
}

export interface TextSupportRequest {
  text: string;
  session_id?: string;
  student_context?: Record<string, unknown>;
  conversation_history?: ConversationTurn[];
}

export interface RiskPayload {
  level?: "low" | "medium" | "high" | "critical" | string;
  score?: number;
  reason?: string;
  trigger_terms?: string[];
  needs_human_followup?: boolean;
}

export interface EntropyPayload {
  score?: number;
  level?: number;
  balance_state?: string;
  dominant_drivers?: string[];
  trend?: {
    previous_score?: number | null;
    delta?: number | null;
    direction?: string;
  };
}

export interface SafetyPayload {
  disclaimer?: string;
  emergency_notice?: string | null;
  human_referral?: string | null;
}

export interface EntropyReductionPayload {
  target_state?: string;
  rationale?: string;
  core_actions?: string[];
  expected_delta_score?: number;
  review_window_hours?: number;
}

export interface CampusResourcePayload {
  resource_id?: string;
  title?: string;
  category?: string;
  summary?: string;
  recommended_actions?: string[];
  relevance_reason?: string;
}

export interface SupportResponse {
  response_id: string;
  reply_text: string;
  risk?: RiskPayload;
  entropy?: EntropyPayload;
  entropy_reduction?: EntropyReductionPayload;
  safety?: SafetyPayload;
  campus_resources?: CampusResourcePayload[];
  referral_decision?: Record<string, unknown>;
  intervention_strategy?: Record<string, unknown>;
  dynamic_adjustment?: Record<string, unknown>;
  state_profile?: Record<string, unknown>;
  system_flags?: Record<string, unknown>;
  session?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface HumanInterventionRequest {
  response_id?: string;
  status: HumanInterventionStatus;
  handler_id?: string;
  note?: string;
  next_action?: string;
  tags?: string[];
}

export interface CareQueueResponse {
  items: unknown[];
  priority_counts: Record<string, number>;
  [key: string]: unknown;
}

export interface StudentDisplayModel {
  replyText: string;
  riskLevel?: string;
  entropyScore?: number;
  balanceState?: string;
  coreActions: string[];
  emergencyNotice?: string | null;
  humanReferral?: string | null;
  campusResources: CampusResourcePayload[];
  raw: SupportResponse;
}

export class CampusSupportApiError extends Error {
  status: number;
  detail: unknown;

  constructor(status: number, detail: unknown) {
    super(typeof detail === "string" ? detail : `Campus support API request failed: ${status}`);
    this.name = "CampusSupportApiError";
    this.status = status;
    this.detail = detail;
  }
}

export class CampusSupportApi {
  private readonly baseUrl: string;

  constructor(baseUrl = "http://127.0.0.1:8000") {
    this.baseUrl = baseUrl.replace(/\/$/, "");
  }

  async health(): Promise<Record<string, unknown>> {
    return this.requestJson("/health");
  }

  async frontendContract(): Promise<Record<string, unknown>> {
    return this.requestJson("/api/v1/frontend/contract");
  }

  async opsReadiness(): Promise<Record<string, unknown>> {
    return this.requestJson("/api/v1/ops/readiness");
  }

  async sendText(payload: TextSupportRequest): Promise<SupportResponse> {
    return this.requestJson("/api/v1/support/text", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        conversation_history: [],
        student_context: {},
        ...payload,
      }),
    });
  }

  async sendAudio(params: {
    file: File;
    session_id?: string;
    student_context?: Record<string, unknown>;
    conversation_history?: ConversationTurn[];
  }): Promise<SupportResponse> {
    const form = new FormData();
    form.append("file", params.file);
    if (params.session_id) form.append("session_id", params.session_id);
    if (params.student_context) form.append("student_context", JSON.stringify(params.student_context));
    if (params.conversation_history) form.append("conversation_history", JSON.stringify(params.conversation_history));
    return this.requestJson("/api/v1/support/audio", { method: "POST", body: form });
  }

  async getSession(sessionId: string): Promise<Record<string, unknown>> {
    return this.requestJson(`/api/v1/sessions/${encodeURIComponent(sessionId)}`);
  }

  async getRoleView(sessionId: string, role: ViewRole = "student"): Promise<Record<string, unknown>> {
    return this.requestJson(`/api/v1/sessions/${encodeURIComponent(sessionId)}/view?role=${encodeURIComponent(role)}`);
  }

  async getCareQueue(options: { includeLowPriority?: boolean; includeResolved?: boolean; limit?: number } = {}): Promise<CareQueueResponse> {
    const query = new URLSearchParams();
    if (options.includeLowPriority !== undefined) query.set("include_low_priority", String(options.includeLowPriority));
    if (options.includeResolved !== undefined) query.set("include_resolved", String(options.includeResolved));
    if (options.limit !== undefined) query.set("limit", String(options.limit));
    const suffix = query.toString() ? `?${query.toString()}` : "";
    return this.requestJson(`/api/v1/analytics/care-queue${suffix}`);
  }

  async appendHumanIntervention(sessionId: string, payload: HumanInterventionRequest): Promise<Record<string, unknown>> {
    return this.requestJson(`/api/v1/sessions/${encodeURIComponent(sessionId)}/human-interventions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  }

  async clearSession(sessionId: string): Promise<Record<string, unknown>> {
    return this.requestJson(`/api/v1/sessions/${encodeURIComponent(sessionId)}`, { method: "DELETE" });
  }

  toStudentDisplayModel(response: SupportResponse): StudentDisplayModel {
    return {
      replyText: response.reply_text,
      riskLevel: response.risk?.level,
      entropyScore: response.entropy?.score,
      balanceState: response.entropy?.balance_state,
      coreActions: response.entropy_reduction?.core_actions || [],
      emergencyNotice: response.safety?.emergency_notice,
      humanReferral: response.safety?.human_referral,
      campusResources: response.campus_resources || [],
      raw: response,
    };
  }

  private async requestJson<T>(path: string, init: RequestInit = {}): Promise<T> {
    const response = await fetch(`${this.baseUrl}${path}`, init);
    const data = await response.json().catch(() => null);
    if (!response.ok) {
      const detail = data && typeof data === "object" && "detail" in data ? data.detail : data;
      throw new CampusSupportApiError(response.status, detail);
    }
    return data as T;
  }
}

export const campusSupportApi = new CampusSupportApi();
