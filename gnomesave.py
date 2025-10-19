import os
import pandas as pd
from datetime import datetime
from typing import List, Dict
import gspread
from google.oauth2.service_account import Credentials
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import HumanMessage, AIMessage
import json
import matplotlib.pyplot as plt
import io
from dotenv import load_dotenv

# IMPORTS ADICIONAIS PARA TRANSCRIÇÃO DE ÁUDIO
from google import genai
from google.genai import types
from google.genai.errors import APIError

load_dotenv()

# ==================== CONFIG ====================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_SHEET_ID = "1JNktH_iNhelblDEmF0GPc3P94Vvxt8nVq0fklbJgkfs"
GOOGLE_CREDENTIALS_PATH = os.getenv("GOOGLE_CREDENTIALS_PATH", "credentials.json")

# ==================== GOOGLE SHEETS CLIENT ====================
def get_google_sheets_client():
    scopes = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]
    creds = Credentials.from_service_account_file(GOOGLE_CREDENTIALS_PATH, scopes=scopes)
    return gspread.authorize(creds)

# ==================== DATA PROCESSOR ====================
class FinanceDataProcessor:
    def __init__(self, sheet_id: str):
        self.sheet_id = sheet_id
        self.client = get_google_sheets_client()
    
    def get_transactions(self) -> pd.DataFrame:
        try:
            sheet = self.client.open_by_key(self.sheet_id)
            worksheet = sheet.worksheet("extrato")
            data = worksheet.get_all_values()
            header_row = None
            for idx, row in enumerate(data):
                if "Data e hora" in row:
                    header_row = idx
                    break
            if header_row is None:
                raise ValueError("Cabeçalho não encontrado")
            df = pd.DataFrame(data[header_row + 1:], columns=data[header_row])
            df['Valor'] = df['Valor'].str.replace('R$', '').str.replace('.', '').str.replace(',', '.').astype(float)
            df['Data e hora'] = pd.to_datetime(df['Data e hora'], errors='coerce')
            return df
        except Exception as e:
            print(f"Erro ao buscar dados: {e}")
            return pd.DataFrame()
    
    def get_expenses_only(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[df['Valor'] < 0].copy()
    
    def analyze_categories(self, df: pd.DataFrame) -> Dict:
        expenses = self.get_expenses_only(df)
        category_analysis = expenses.groupby('Categoria').agg({
            'Valor': ['sum', 'count', 'mean']
        }).round(2)
        category_analysis.columns = ['total', 'count', 'avg']
        category_analysis['percentage'] = (
            category_analysis['total'] / category_analysis['total'].sum() * 100
        ).round(2)
        return category_analysis.sort_values('total').to_dict('index')
    
    def detect_recurring_expenses(self, df: pd.DataFrame) -> List[Dict]:
        expenses = self.get_expenses_only(df)
        recurring = []
        for desc, group in expenses.groupby('Descrição'):
            if len(group) >= 2:
                values = group['Valor'].abs()
                if values.std() / values.mean() < 0.1:
                    recurring.append({
                        'merchant': desc,
                        'avg_amount': values.mean(),
                        'count': len(group),
                        'category': group['Categoria'].iloc[0],
                        'frequency': self._calculate_frequency(group['Data e hora'])
                    })
        return recurring
    
    def _calculate_frequency(self, dates: pd.Series) -> str:
        if len(dates) < 2:
            return "irregular"
        dates_sorted = dates.sort_values()
        avg_days = (dates_sorted.max() - dates_sorted.min()).days / (len(dates) - 1)
        if avg_days < 10:
            return "semanal"
        elif avg_days < 35:
            return "mensal"
        elif avg_days < 100:
            return "trimestral"
        else:
            return "anual"
    
    def get_summary_stats(self, df: pd.DataFrame) -> Dict:
        """Gera estatísticas resumidas ao invés de enviar todos os dados"""
        expenses = self.get_expenses_only(df)
        
        # Agrupa por categoria
        by_category = expenses.groupby('Categoria').agg({
            'Valor': ['sum', 'count', 'mean']
        }).round(2)
        by_category.columns = ['total', 'count', 'avg']
        by_category['total'] = by_category['total'].abs()
        by_category['avg'] = by_category['avg'].abs()
        
        # Top merchants
        top_merchants = expenses.groupby('Descrição')['Valor'].sum().abs().nlargest(10)
        
        # Gastos por mês
        expenses['mes'] = expenses['Data e hora'].dt.to_period('M')
        monthly = expenses.groupby('mes')['Valor'].sum().abs()
        
        return {
            'total_gasto': float(expenses['Valor'].abs().sum()),
            'num_transacoes': len(expenses),
            'ticket_medio': float(expenses['Valor'].abs().mean()),
            'por_categoria': by_category.to_dict('index'),
            'top_10_merchants': top_merchants.to_dict(),
            'gastos_mensais': {str(k): float(v) for k, v in monthly.items()},
            'periodo': {
                'inicio': str(expenses['Data e hora'].min().date()),
                'fim': str(expenses['Data e hora'].max().date())
            }
        }
    
    def generate_category_pie_chart(self, df: pd.DataFrame) -> io.BytesIO:
        expenses = self.get_expenses_only(df)
        expenses['Valor_Abs'] = expenses['Valor'].abs()
        gastos_por_categoria = expenses.groupby('Categoria')['Valor_Abs'].sum().sort_values(ascending=False)
        valores = gastos_por_categoria.values
        categorias = gastos_por_categoria.index.tolist()
        cores = ['#03ef62', '#06bdfc', '#ff6ea9', '#ff931e', '#ff5400', '#7933ff', '#00d4aa', '#ffd700', '#ff69b4', '#8a2be2', '#32cd32', '#ff4500']
        plt.figure(figsize=(12, 8))
        plt.pie(x=valores, labels=categorias, colors=cores[:len(categorias)], autopct='%1.1f%%', pctdistance=0.8, labeldistance=1.1, startangle=180)
        plt.title('Distribuição de Gastos por Categoria', fontsize=16, pad=20)
        plt.legend(labels=[f'{cat}: R$ {val:,.2f}' for cat, val in zip(categorias, valores)], loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=200, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        return buf

# ==================== AUDIO TRANSCRIBER (NOVO) ====================
class AudioTranscriber:
    """Lida com o download e a transcrição do áudio usando o Gemini SDK nativo."""
    def __init__(self, api_key: str):
        # Inicializa o cliente GenAI nativo para lidar com uploads e transcrição
        self.client = genai.Client(api_key=api_key)

    async def transcribe_audio(self, context: ContextTypes.DEFAULT_TYPE, file_id: str) -> str:
        """Faz o download do áudio, carrega no Gemini e transcreve."""
        
        # 1. Obter o objeto File do Telegram
        new_file = await context.bot.get_file(file_id)
        
        # 2. Fazer o download do áudio em um arquivo temporário
        temp_audio_path = f"temp_audio_{file_id}.ogg" 
        await new_file.download_to_drive(custom_path=temp_audio_path)
        
        transcription = "Erro na transcrição do áudio."
        audio_file_uploaded = None
        
        try:
            # 3. Upload do arquivo para o Gemini
            audio_file_uploaded = self.client.files.upload(file=temp_audio_path)
            
            # 4. Transcrição usando o Gemini
            prompt = "Transcreva este áudio completamente."
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[audio_file_uploaded, prompt]
            )
            
            transcription = response.text
            
        except APIError as e:
            print(f"Erro na API do Gemini durante a transcrição: {e}")
            transcription = "Erro ao processar o áudio com Gemini. Tente novamente."
        except Exception as e:
            print(f"Erro inesperado durante a transcrição: {e}")
        finally:
            # 5. Limpeza: Deleta o arquivo temporário e o arquivo no servidor do Gemini
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            if audio_file_uploaded:
                try:
                    self.client.files.delete(name=audio_file_uploaded.name)
                except Exception as e:
                    print(f"Erro ao deletar arquivo do Gemini: {e}")

        return transcription
# =============================================================

# ==================== CONVERSATION MANAGER ====================
class ConversationManager:
    """Guarda histórico de mensagens e contexto de dados para cada usuário"""
    def __init__(self):
        self.histories = {}
        self.last_data_context = {}  # Armazena o último contexto de dados usado
    
    def add_message(self, user_id: int, role: str, content: str):
        if user_id not in self.histories:
            self.histories[user_id] = []
        self.histories[user_id].append({"role": role, "content": content})
        # Keep last 15 messages for better context
        self.histories[user_id] = self.histories[user_id][-15:]
    
    def get_history(self, user_id: int) -> List[Dict]:
        return self.histories.get(user_id, [])
    
    def get_formatted_history(self, user_id: int) -> str:
        """Retorna o histórico formatado para o LLM"""
        history = self.get_history(user_id)
        if not history:
            return ""
        
        formatted = "HISTÓRICO DA CONVERSA:\n"
        for msg in history[-6:]:  # Últimas 6 mensagens
            role = "USUÁRIO" if msg['role'] == "user" else "ASSISTENTE"
            formatted += f"{role}: {msg['content'][:200]}...\n" if len(msg['content']) > 200 else f"{role}: {msg['content']}\n"
        return formatted
    
    def set_data_context(self, user_id: int, context: Dict):
        """Armazena o contexto de dados da última análise"""
        self.last_data_context[user_id] = context
    
    def get_data_context(self, user_id: int) -> Dict:
        """Recupera o contexto de dados da última análise"""
        return self.last_data_context.get(user_id, {})

# ==================== LLM ANALYZER ====================
class LLMFinanceAnalyzer:
    def __init__(self, api_key: str):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.7
        )
    
    def classify_message(self, message: str, conversation_history: str) -> str:
        """Classifica a mensagem considerando o contexto da conversa"""
        prompt = PromptTemplate(
    input_variables=["message", "history"],
    template="""Analise a mensagem do usuário e classifique em uma destas categorias:

CATEGORIAS DE ANÁLISE (exigem dados financeiros):
- analise_cat_gastos: quando pede análise de categorias, distribuição de gastos
- gastos_desnecessarios: quando pergunta sobre gastos desnecessários ou supérfluos
- sugest_economizar: quando pede dicas para economizar
- comp_gastos_mensais: quando quer comparar gastos entre meses

CATEGORIAS CONVERSACIONAIS (NÃO exigem dados novos):
- greeting: saudações como "oi", "olá", "bom dia", "como vai"
- capabilities: perguntas sobre o que o bot pode fazer, suas funcionalidades
- followup: perguntas de acompanhamento, pedidos de explicação sobre algo mencionado anteriormente
- conceito: pedidos para explicar termos ou conceitos (ex: "o que são gastos zumbis?")

CATEGORIA INVÁLIDA:
- invalida: assuntos não relacionados a finanças

{history}

MENSAGEM ATUAL: {message}

IMPORTANTE: Se o usuário está pedindo explicação sobre algo mencionado na conversa anterior, classifique como "followup".

Responda APENAS com a categoria. Seja conciso."""
)
        chain = LLMChain(llm=self.llm, prompt=prompt)
        result = chain.run(message=message, history=conversation_history).strip().lower()
        
        valid_categories = [
            'analise_cat_gastos', 'gastos_desnecessarios', 'sugest_economizar', 
            'comp_gastos_mensais', 'greeting', 'capabilities', 'followup', 'conceito', 'invalida'
        ]
        
        for cat in valid_categories:
            if cat in result:
                return cat
        return 'invalida'
    
    def answer_followup(self, user_message: str, conversation_history: str, data_context: Dict) -> str:
        """Responde perguntas de acompanhamento usando o contexto da conversa"""
        prompt = PromptTemplate(
    input_variables=["message", "history", "context"],
    template="""Você é um assistente financeiro conversando com o usuário.

{history}

CONTEXTO DE DADOS (se disponível):
{context}

PERGUNTA ATUAL DO USUÁRIO:
{message}

Responda de forma natural e educativa, usando exemplos e analogias se possível.
Resuma sua resposta em no máximo 2 parágrafos."""
)
        
        context_str = json.dumps(data_context, ensure_ascii=False) if data_context else "Nenhum dado financeiro carregado"
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        response = chain.run(
            message=user_message,
            history=conversation_history,
            context=context_str
        )
        
        return self._extract_response(response)
    
    def analyze_spending_patterns(self, summary_stats: Dict, user_question: str, conversation_history: str = "") -> str:
        prompt = PromptTemplate(
    input_variables=["stats", "question", "history"],
    template="""Você é especialista em análise financeira.

{history}

RESUMO FINANCEIRO:
{stats}

PERGUNTA DO USUÁRIO:
{question}

Analise padrões principais: categorias mais impactantes, possíveis vazamentos financeiros, gastos recorrentes ou elevados e oportunidades de economia.
Responda em no máximo 2 parágrafos concisos."""
)
        chain = LLMChain(llm=self.llm, prompt=prompt)
        response = chain.run(
            stats=json.dumps(summary_stats, ensure_ascii=False, indent=2), 
            question=user_question,
            history=conversation_history
        )
        return self._extract_response(response)
    
    def analyze_unnecessary_spending(self, data: Dict, conversation_history: str = "") -> str:
        prompt = PromptTemplate(
    input_variables=["recurring", "categories", "history"],
    template="""Analise estes dados e identifique gastos desnecessários:

{history}

GASTOS RECORRENTES: {recurring}
CATEGORIAS: {categories}

Seja específico e prático. Resuma sua resposta em no máximo 2 parágrafos."""
)
        chain = LLMChain(llm=self.llm, prompt=prompt)
        response = chain.run(
            recurring=json.dumps(data.get('recurring', []), ensure_ascii=False),
            categories=json.dumps(data.get('categories', {}), ensure_ascii=False),
            history=conversation_history
        )
        return self._extract_response(response)
    
    def _extract_response(self, response) -> str:
        """Extrai o texto da resposta independente do formato"""
        if isinstance(response, AIMessage):
            return response.content
        elif isinstance(response, list):
            return "\n".join(r.content for r in response if isinstance(r, AIMessage))
        else:
            return str(response)

# ==================== TELEGRAM BOT ====================
class FinanceBot:
    def __init__(self):
        self.data_processor = FinanceDataProcessor(GOOGLE_SHEET_ID)
        self.llm_analyzer = LLMFinanceAnalyzer(GOOGLE_API_KEY)
        self.conv_manager = ConversationManager()
        # Inicializa o Transcritor de Áudio (NOVO)
        self.audio_transcriber = AudioTranscriber(GOOGLE_API_KEY) 
    
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        welcome_msg = """🏦 Olá! Sou seu assistente financeiro pessoal!

Comandos disponíveis:
/analise - Análise de categorias de gasto
/grafico - Gráfico de pizza
/desnecessarios - Gastos desnecessários
/dicas - Sugestões para economizar
/comparar - Comparar gastos mensais

Ou envie uma mensagem sobre suas finanças (pode ser **áudio ou texto**)! Você também pode fazer perguntas de acompanhamento sobre qualquer conceito que eu mencionar. 💬"""
        await update.message.reply_text(welcome_msg)
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.message.from_user.id
        user_message = update.message.text
        
        # ==========================================================
        # NOVO: Lida com a transcrição de áudio no começo do pipeline
        if update.message.voice:
            await update.message.reply_text("🎙️ Áudio recebido! Transcrevendo com Gemini...")
            try:
                # Obtém a transcrição e a usa como a 'mensagem' do usuário
                audio_file_id = update.message.voice.file_id
                transcribed_text = await self.audio_transcriber.transcribe_audio(context, audio_file_id)
                
                if "Erro" in transcribed_text: # Se houve um erro na transcrição
                    await update.message.reply_text(f"❌ Erro na transcrição: {transcribed_text}")
                    return
                
                user_message = transcribed_text
                await update.message.reply_text(f"📝 **Transcrição**: \n*'{user_message[:200]}{'...' if len(user_message) > 200 else ''}'*")
                
            except Exception as e:
                print(f"Erro ao processar áudio: {e}")
                await update.message.reply_text("❌ Desculpe, houve um erro ao processar seu áudio.")
                return

        # Se for um áudio vazio ou outro tipo de mídia que não foi transcrita, ignora
        if not user_message:
            return
        # ==========================================================

        # Adiciona mensagem do usuário (transcrita ou texto) ao histórico
        self.conv_manager.add_message(user_id, "user", user_message)
        
        # Obtém histórico formatado
        conversation_history = self.conv_manager.get_formatted_history(user_id)
        
        # Classifica a mensagem considerando o contexto
        category = self.llm_analyzer.classify_message(user_message, conversation_history)
        
        # Mensagens inválidas
        if category == 'invalida':
            response = "Me desculpe, eu só posso responder perguntas sobre finanças. 💰"
            await update.message.reply_text(response)
            self.conv_manager.add_message(user_id, "assistant", response)
            return
        
        # Saudações
        if category == 'greeting':
            response = """Olá! 👋 Prazer em te conhecer!

Sou seu assistente financeiro pessoal. Posso te ajudar a:
• 📊 Analisar seus gastos por categoria
• 💸 Identificar gastos desnecessários
• 💡 Dar dicas para economizar
• 📈 Comparar gastos entre meses
• 🎯 Detectar gastos recorrentes

Como posso te ajudar hoje?"""
            await update.message.reply_text(response)
            self.conv_manager.add_message(user_id, "assistant", response)
            return
        
        # Perguntas sobre capacidades
        if category == 'capabilities':
            response = """Posso fazer várias coisas para te ajudar com suas finanças! 🤓

📊 **Análises**:
• Distribuição de gastos por categoria (com gráfico!)
• Identificação de gastos desnecessários
• Detecção de gastos recorrentes ("gastos zumbis")

💡 **Dicas práticas**:
• Sugestões personalizadas para economizar
• Análise de padrões de consumo
• Comparação entre períodos

🗣️ **Conversação**:
• Posso explicar conceitos financeiros
• Responder dúvidas sobre suas análises
• Ter uma conversa fluida sobre suas finanças

Basta me perguntar algo sobre suas finanças ou usar um dos comandos do menu!"""
            await update.message.reply_text(response)
            self.conv_manager.add_message(user_id, "assistant", response)
            return
        
        # Perguntas de acompanhamento ou conceituais
        if category in ['followup', 'conceito']:
            await update.message.reply_text("🤔 Deixa eu pensar...")
            data_context = self.conv_manager.get_data_context(user_id)
            analysis = self.llm_analyzer.answer_followup(
                user_message, 
                conversation_history,
                data_context
            )
            self.conv_manager.add_message(user_id, "assistant", analysis)
            await self.send_long_message(update, analysis)
            return
        
        # Análises que exigem dados financeiros
        await update.message.reply_text("📊 Analisando seus dados financeiros...")
        df = self.data_processor.get_transactions()
        
        if df.empty:
            response = "❌ Não consegui acessar seus dados financeiros. Verifique a planilha."
            await update.message.reply_text(response)
            self.conv_manager.add_message(user_id, "assistant", response)
            return

        # Gera resumo estatístico ao invés de passar todos os dados
        summary_stats = self.data_processor.get_summary_stats(df)
        
        # Armazena contexto de dados
        data_context = {
            'total_transactions': len(df),
            'total_expenses': summary_stats['total_gasto'],
            'categories': list(df['Categoria'].unique()),
            'summary': summary_stats
        }
        self.conv_manager.set_data_context(user_id, data_context)

        # Processa de acordo com a categoria
        if category == 'analise_cat_gastos':
            # Gera o gráfico
            chart_buffer = self.data_processor.generate_category_pie_chart(df)
            await update.message.reply_photo(
                photo=chart_buffer, 
                caption="📊 Distribuição de Gastos por Categoria"
            )
            # Gera a análise textual
            await update.message.reply_text("✍️ Preparando análise detalhada...")
            analysis = self.llm_analyzer.analyze_spending_patterns(
                summary_stats,
                user_message,
                conversation_history
            )
        elif category == 'gastos_desnecessarios':
            recurring = self.data_processor.detect_recurring_expenses(df)
            categories = self.data_processor.analyze_categories(df)
            data_context.update({'recurring': recurring, 'categories': categories})
            self.conv_manager.set_data_context(user_id, data_context)
            analysis = self.llm_analyzer.analyze_unnecessary_spending(
                {'recurring': recurring, 'categories': categories},
                conversation_history
            )
        else:
            analysis = self.llm_analyzer.analyze_spending_patterns(
                summary_stats,
                user_message,
                conversation_history
            )
        
        self.conv_manager.add_message(user_id, "assistant", analysis)
        await self.send_long_message(update, analysis)
    
    async def handle_chart_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("📊 Gerando gráfico...")
        df = self.data_processor.get_transactions()
        if df.empty:
            await update.message.reply_text("❌ Não consegui acessar seus dados financeiros.")
            return
        chart_buffer = self.data_processor.generate_category_pie_chart(df)
        expenses = self.data_processor.get_expenses_only(df)
        total = expenses['Valor'].abs().sum()
        await update.message.reply_photo(
            photo=chart_buffer, 
            caption=f"📊 Distribuição de Gastos\n💰 Total: R$ {total:,.2f}"
        )

    async def send_long_message(self, update: Update, text: str):
        max_len = 4000
        for i in range(0, len(text), max_len):
            await update.message.reply_text(text[i:i+max_len])

    def run(self):
        application = Application.builder().token(TELEGRAM_TOKEN).build()
        application.add_handler(CommandHandler("start", self.start))
        application.add_handler(CommandHandler("grafico", self.handle_chart_command))
        
        # Adiciona o handler para mensagens de texto E áudio (MODIFICADO)
        text_and_voice_handler = MessageHandler(
            (filters.TEXT & ~filters.COMMAND) | filters.VOICE, 
            self.handle_message
        )
        application.add_handler(text_and_voice_handler)
        
        print("🤖 Bot iniciado com capacidade conversacional e transcrição de áudio!")
        application.run_polling()

# ==================== MAIN ====================
if __name__ == "__main__":
    bot = FinanceBot()
    bot.run()