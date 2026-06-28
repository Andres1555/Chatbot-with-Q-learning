# Interfaz grafica (Tkinter) del chatbot UNEG con Q-Learning
# Panel izquierdo: chat con botones Like/Dislike
# Panel derecho: tabla Q, formula de Bellman y ultima actualizacion
import tkinter as tk
from tkinter import ttk
from src.dataset import STATE_NAMES, ALL_RESPONSES, TOTAL_ACTIONS
from src.classifier import IntentClassifier
from src.agent import QLearningAgent


class ChatbotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("UNEG - Chatbot con Q-Learning")
        self.root.geometry("1100x700")
        self.root.minsize(900, 600)

        # Instancia el clasificador y el agente (se reinician al cerrar)
        self.classifier = IntentClassifier()
        self.agent = QLearningAgent()

        self.last_state = None
        self.last_action = None

        self._setup_styles()
        self._build_ui()

        # Mensaje de bienvenida
        self.add_bot_message("Hola! Soy el asistente virtual de la UNEG. "
                             "Preguntame sobre carreras, inscripciones, requisitos u horarios. "
                             "Cada vez que responda, puedes calificar mi respuesta con Like o Dislike "
                             "para que yo aprenda a responder mejor. Empecemos!")

    # Configura el estilo visual de la UI
    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#f5f5f5")
        style.configure("TLabel", background="#f5f5f5", font=("Segoe UI", 10))
        style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"), foreground="#1a73e8")
        style.configure("Status.TLabel", font=("Segoe UI", 9), foreground="#555555")
        style.configure("Send.TButton", font=("Segoe UI", 10, "bold"), background="#1a73e8", foreground="white")
        style.map("Send.TButton", background=[("active", "#1557b0")])

    # Construye la interfaz completa (header, paneles, status bar)
    def _build_ui(self):
        self.root.rowconfigure(0, weight=0)
        self.root.rowconfigure(1, weight=1)
        self.root.columnconfigure(0, weight=1)

        # Header con titulo
        header = ttk.Frame(self.root, style="TFrame")
        header.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))
        ttk.Label(header, text="UNEG", style="Header.TLabel").pack(side="left")
        ttk.Label(header, text="Chatbot con Aprendizaje por Refuerzo (Q-Learning)",
                  font=("Segoe UI", 10), foreground="#666666").pack(side="left", padx=(15, 0))

        # Panel dividido: chat (izquierda) e info (derecha)
        paned = ttk.PanedWindow(self.root, orient="horizontal")
        paned.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)

        self._build_chat_panel(paned)
        self._build_info_panel(paned)

        # Barra de estado inferior
        status_frame = ttk.Frame(self.root, style="TFrame")
        status_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 8))
        self.status_label = ttk.Label(status_frame, text="Listo", style="Status.TLabel")
        self.status_label.pack(side="left")
        self.stats_label = ttk.Label(status_frame, text="", style="Status.TLabel")
        self.stats_label.pack(side="right")
        self._update_stats()

    # Panel izquierdo: historial del chat + entrada de texto
    def _build_chat_panel(self, paned):
        chat_frame = ttk.Frame(paned, style="TFrame")
        paned.add(chat_frame, weight=3)
        chat_frame.rowconfigure(0, weight=1)
        chat_frame.rowconfigure(1, weight=0)
        chat_frame.columnconfigure(0, weight=1)

        # Canvas con scroll para los mensajes
        self.chat_canvas = tk.Canvas(chat_frame, bg="#ffffff", highlightthickness=1,
                                     highlightbackground="#e0e0e0", relief="flat")
        self.chat_scrollbar = ttk.Scrollbar(chat_frame, orient="vertical", command=self.chat_canvas.yview)
        self.chat_canvas.configure(yscrollcommand=self.chat_scrollbar.set)
        self.chat_canvas.grid(row=0, column=0, sticky="nsew")
        self.chat_scrollbar.grid(row=0, column=1, sticky="ns")

        # Frame interno dentro del canvas
        self.chat_inner = ttk.Frame(self.chat_canvas, style="TFrame")
        self.chat_window = self.chat_canvas.create_window(
            (0, 0), window=self.chat_inner, anchor="nw",
            width=self.chat_canvas.winfo_reqwidth()
        )
        self.chat_inner.bind("<Configure>", self._on_chat_configure)
        self.chat_canvas.bind("<Configure>", self._on_canvas_configure)

        # Entrada de texto y boton enviar
        input_frame = ttk.Frame(chat_frame, style="TFrame")
        input_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        input_frame.columnconfigure(0, weight=1)

        self.entry_var = tk.StringVar()
        self.entry = ttk.Entry(input_frame, textvariable=self.entry_var, font=("Segoe UI", 11))
        self.entry.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.entry.bind("<Return>", self._send_message)

        self.send_btn = ttk.Button(input_frame, text="Enviar", style="Send.TButton",
                                   command=self._send_message, width=12)
        self.send_btn.grid(row=0, column=1, sticky="e")

    # Panel derecho: formula de Bellman, tabla Q y ultima actualizacion
    def _build_info_panel(self, paned):
        info_frame = ttk.Frame(paned, style="TFrame")
        paned.add(info_frame, weight=2)
        info_frame.rowconfigure(0, weight=0)
        info_frame.rowconfigure(1, weight=1)
        info_frame.rowconfigure(2, weight=1)
        info_frame.columnconfigure(0, weight=1)

        # Seccion de formula de Bellman
        bellman_frame = ttk.LabelFrame(info_frame, text="Ecuacion de Bellman", padding=10)
        bellman_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        bellman_frame.columnconfigure(0, weight=1)

        self.bellman_text = tk.Text(bellman_frame, height=10, width=30, font=("Consolas", 10),
                                    bg="#ffffff", fg="#333333", relief="flat", wrap="word",
                                    highlightthickness=1, highlightbackground="#e0e0e0")
        self.bellman_text.pack(fill="both", expand=True)
        self._update_bellman_display()

        # Seccion de tabla Q (estados x acciones)
        q_frame = ttk.LabelFrame(info_frame, text="Tabla Q (Estados x Acciones)", padding=10)
        q_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 8))
        q_frame.rowconfigure(0, weight=1)
        q_frame.columnconfigure(0, weight=1)

        q_container = ttk.Frame(q_frame)
        q_container.grid(row=0, column=0, sticky="nsew")
        q_container.rowconfigure(0, weight=1)
        q_container.columnconfigure(0, weight=1)

        self.q_tree = ttk.Treeview(q_container, show="headings", height=8)
        self.q_tree_scroll = ttk.Scrollbar(q_container, orient="vertical", command=self.q_tree.yview)
        self.q_tree.configure(yscrollcommand=self.q_tree_scroll.set)
        self.q_tree.grid(row=0, column=0, sticky="nsew")
        self.q_tree_scroll.grid(row=0, column=1, sticky="ns")
        self._update_q_table()

        # Seccion de ultima actualizacion
        hist_frame = ttk.LabelFrame(info_frame, text="Ultima Actualizacion", padding=10)
        hist_frame.grid(row=2, column=0, sticky="nsew")
        hist_frame.columnconfigure(0, weight=1)

        self.update_text = tk.Text(hist_frame, height=6, width=30, font=("Consolas", 9),
                                   bg="#fafafa", fg="#333333", relief="flat", wrap="word",
                                   highlightthickness=1, highlightbackground="#e0e0e0")
        self.update_text.pack(fill="both", expand=True)
        self._update_last_update()

    # Ajusta el area de scroll cuando el contenido cambia
    def _on_chat_configure(self, event):
        self.chat_canvas.configure(scrollregion=self.chat_canvas.bbox("all"))

    # Ajusta el ancho del frame interno cuando el canvas cambia de tamano
    def _on_canvas_configure(self, event):
        self.chat_canvas.itemconfig(self.chat_window, width=event.width)

    # Maneja el envio de un mensaje del usuario
    def _send_message(self, event=None):
        msg = self.entry_var.get().strip()
        if not msg:
            return
        self.entry_var.set("")
        self.add_user_message(msg)
        self.status_label.config(text="Procesando...")
        self.root.update_idletasks()
        # Pequena pausa para que se vea el "Procesando..."
        self.root.after(300, lambda: self._process_and_respond(msg))

    # Procesa el mensaje: clasifica, selecciona accion y muestra respuesta
    def _process_and_respond(self, msg):
        state_key = self.classifier.classify(msg)
        action_idx = self.agent.select_action(state_key)
        response = ALL_RESPONSES[action_idx]["text"]
        origin = ALL_RESPONSES[action_idx]["origin_intent"]

        self.last_state = state_key
        self.last_action = action_idx

        self.add_bot_message(response, state_key, action_idx, origin)
        self.status_label.config(
            text=f"Intencion: {state_key} | Accion: A{action_idx} (original: {origin})"
        )
        self._update_stats()
        self._update_bellman_display()

    # Agrega un mensaje del usuario al chat 
    def add_user_message(self, msg):
        frame = ttk.Frame(self.chat_inner, style="TFrame")
        frame.pack(fill="x", padx=10, pady=4)

        bubble = tk.Frame(frame, bg="#1a73e8", padx=12, pady=8)
        bubble.pack(side="right", anchor="e")

        label = tk.Label(bubble, text=msg, font=("Segoe UI", 10),
                         bg="#1a73e8", fg="white", wraplength=380, justify="left")
        label.pack()

        self.root.after(50, self._scroll_to_bottom)

    # Agrega un mensaje del bot al chat 
    # Incluye botones Like/Dislike para calificar la respuesta
    def add_bot_message(self, msg, state_key=None, action_idx=None, origin=None):
        frame = ttk.Frame(self.chat_inner, style="TFrame")
        frame.pack(fill="x", padx=10, pady=4)

        bubble = tk.Frame(frame, bg="#f0f0f0", padx=12, pady=8, highlightthickness=1,
                          highlightbackground="#e0e0e0")
        bubble.pack(side="left", anchor="w")

        label = tk.Label(bubble, text=msg, font=("Segoe UI", 10),
                         bg="#f0f0f0", fg="#333333", wraplength=380, justify="left")
        label.pack(anchor="w")

        # Muestra la accion y el origen si estan disponibles
        if state_key is not None and action_idx is not None:
            info_text = f"A{action_idx} (original: {origin})"
            info_label = tk.Label(bubble, text=info_text, font=("Consolas", 7),
                                  bg="#f0f0f0", fg="#999999")
            info_label.pack(anchor="w", pady=(2, 4))

            # Frame para botones Like/Dislike
            btn_frame = tk.Frame(bubble, bg="#f0f0f0")
            btn_frame.pack(anchor="w", pady=(0, 0))

            like_btn = tk.Button(
                btn_frame, text="  Like (+1)  ", font=("Segoe UI", 9, "bold"),
                bg="#e8f5e9", fg="#2e7d32", relief="flat", padx=6, pady=2,
                activebackground="#c8e6c9", cursor="hand2"
            )
            dislike_btn = tk.Button(
                btn_frame, text="Dislike (-1)", font=("Segoe UI", 9, "bold"),
                bg="#ffebee", fg="#c62828", relief="flat", padx=6, pady=2,
                activebackground="#ffcdd2", cursor="hand2"
            )

            # Configurar comandos con captura de variables por defecto
            like_btn.config(
                command=lambda s=state_key, a=action_idx, lb=like_btn, db=dislike_btn:
                    self._give_feedback(s, a, 1, lb, db)
            )
            dislike_btn.config(
                command=lambda s=state_key, a=action_idx, lb=like_btn, db=dislike_btn:
                    self._give_feedback(s, a, -1, lb, db)
            )

            like_btn.pack(side="left", padx=(0, 4))
            dislike_btn.pack(side="left")

        self.root.after(50, self._scroll_to_bottom)

    # Procesa el feedback del usuario: actualiza Q, desactiva botones y refresca UI
    def _give_feedback(self, state_key, action_idx, reward, like_btn, dislike_btn):
        # Desactiva los botones para evitar doble click
        like_btn.config(state="disabled", bg="#d0d0d0", fg="#888888", cursor="arrow")
        dislike_btn.config(state="disabled", bg="#d0d0d0", fg="#888888", cursor="arrow")

        # Actualiza el agente Q-Learning con la recompensa
        self.agent.update(state_key, action_idx, reward)
        # Refresca los paneles de informacion
        self._update_q_table()
        self._update_last_update()
        self._update_stats()
        self._update_bellman_display()

        # Actualiza la barra de estado
        reward_text = "Like (+1)" if reward > 0 else "Dislike (-1)"
        origin = self.agent.get_action_origin(action_idx)
        self.status_label.config(
            text=f"Recibido: {reward_text} | Estado: {state_key} | Accion: A{action_idx} (respuesta de: {origin})"
        )

    # Desplaza el chat hacia abajo para mostrar el ultimo mensaje
    def _scroll_to_bottom(self):
        self.chat_canvas.update_idletasks()
        self.chat_canvas.yview_moveto(1.0)

    # Actualiza la tabla Q en la UI
    def _update_q_table(self):
        for item in self.q_tree.get_children():
            self.q_tree.delete(item)

        col_indices, rows = self.agent.get_q_table_display()
        columns = ["state"] + [f"A{a}" for a in col_indices]
        self.q_tree["columns"] = columns

        self.q_tree.heading("state", text="Estado")
        self.q_tree.column("state", width=90, anchor="w", minwidth=70)
        for a in col_indices:
            origin = self.agent.get_action_origin(a)
            col = f"A{a}"
            short = origin[:6] + ".." if len(origin) > 8 else origin
            self.q_tree.heading(col, text=f"A{a}\n{short}")
            self.q_tree.column(col, width=62, anchor="center", minwidth=50)

        for row in rows:
            display_name = STATE_NAMES.get(row["state"], row["state"])
            values = [display_name]
            for a in col_indices:
                qv = row.get(a, 0.0)
                values.append(f"{qv:.3f}")
            self.q_tree.insert("", "end", values=values, tags=(row["state"],))

    # Actualiza el panel de la formula de Bellman
    def _update_bellman_display(self):
        self.bellman_text.config(state="normal")
        self.bellman_text.delete("1.0", "end")
        lines = self.agent.get_bellman_formula()
        for i, line in enumerate(lines):
            self.bellman_text.insert("end", line + "\n")
            if i == 1:
                self.bellman_text.tag_add("formula", f"{i+1}.0", f"{i+1}.end")
                self.bellman_text.tag_config(
                    "formula", font=("Consolas", 11, "bold"), foreground="#d32f2f"
                )

        stats = self.agent.get_stats()
        self.bellman_text.insert("end", f"\nEstadisticas:\n")
        self.bellman_text.insert("end", f"Interacciones: {stats['total']}\n")
        self.bellman_text.insert("end", f"Likes: {stats['likes']} | Dislikes: {stats['dislikes']}\n")
        self.bellman_text.insert("end", f"Precision: {stats['acierto']}%\n")
        self.bellman_text.config(state="disabled")

    # Actualiza el panel de la ultima actualizacion
    def _update_last_update(self):
        self.update_text.config(state="normal")
        self.update_text.delete("1.0", "end")
        text = self.agent.get_last_update_str()
        self.update_text.insert("1.0", text)
        self.update_text.config(state="disabled")

    # Actualiza la barra de estadisticas en la parte inferior
    def _update_stats(self):
        stats = self.agent.get_stats()
        self.stats_label.config(
            text=f"Interacciones: {stats['total']} | "
                 f"Likes: {stats['likes']} | Dislikes: {stats['dislikes']} | "
                 f"Precision: {stats['acierto']}% | "
                 f"epsilon: {self.agent.epsilon:.3f}"
        )
