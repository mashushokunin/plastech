from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "pdf" / "forward_inverse_summary_2026-10-20.pdf"

NAVY = colors.HexColor("#17324D")
BLUE = colors.HexColor("#2E6F95")
PALE_BLUE = colors.HexColor("#EAF3F8")
PALE_GREEN = colors.HexColor("#EAF5EF")
PALE_YELLOW = colors.HexColor("#FFF4D6")
GREY = colors.HexColor("#5F6B73")
LIGHT_GREY = colors.HexColor("#F3F5F6")


def register_fonts() -> None:
    regular = next(
        Path("/System/Library/AssetsV2").glob("**/BIZ_UDMincho-regular.ttf")
    )
    pdfmetrics.registerFont(TTFont("JP", str(regular)))
    pdfmetrics.registerFont(TTFont("JP-Bold", str(regular)))


def p(text: object, style: ParagraphStyle) -> Paragraph:
    return Paragraph(str(text), style)


def table(data, widths, *, font_size=7.5, centered=False) -> Table:
    rows = []
    for row_index, row in enumerate(data):
        style = STYLES["table_header"] if row_index == 0 else STYLES["table_cell"]
        rows.append([p(value, style) for value in row])
    result = Table(rows, colWidths=widths, repeatRows=1, hAlign="CENTER" if centered else "LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD3D8")),
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
    ]
    for row_index in range(2, len(rows), 2):
        commands.append(("BACKGROUND", (0, row_index), (-1, row_index), LIGHT_GREY))
    result.setStyle(TableStyle(commands))
    return result


def card(title: str, body: str, background=PALE_BLUE) -> Table:
    content = [p(title, STYLES["card_title"]), Spacer(1, 2 * mm), p(body, STYLES["body"])]
    result = Table([[content]], colWidths=[174 * mm])
    result.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), background),
                ("BOX", (0, 0), (-1, -1), 0.6, BLUE),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return result


def bullet(text: str) -> Paragraph:
    return Paragraph(f"• {text}", STYLES["bullet"])


def footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D6DDE1"))
    canvas.line(18 * mm, 13 * mm, 192 * mm, 13 * mm)
    canvas.setFont("JP", 7)
    canvas.setFillColor(GREY)
    canvas.drawString(18 * mm, 8 * mm, "順解析・逆解析の手法と結果 | 2026-10-20")
    canvas.drawRightString(192 * mm, 8 * mm, str(doc.page))
    canvas.restoreState()


def build() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = BaseDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=17 * mm,
        bottomMargin=18 * mm,
        title="順解析・逆解析の手法と結果まとめ",
        author="プロジェクトチーム",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="content")
    doc.addPageTemplates([PageTemplate(id="summary", frames=[frame], onPage=footer)])

    story = [
        p("プラス・テク共同検討", STYLES["eyebrow"]),
        p("順解析・逆解析の手法と結果", STYLES["title"]),
        p("前回資料から最新の候補選定までをつなぐ要約 | 2026年10月20日", STYLES["subtitle"]),
        Spacer(1, 8 * mm),
        card(
            "一言で表すと",
            "順解析は、配合から物性を予測する『候補を測る物差し』。逆解析は、その物差しを使って目標に近い配合候補を探し、複数提示する仕組みである。",
            PALE_GREEN,
        ),
        Spacer(1, 7 * mm),
        p("前回資料との関係", STYLES["h2"]),
        table(
            [
                ["時点・資料", "位置づけ", "分かったこと"],
                ["2026-07<br/>順解析・逆解析レポート", "全体の基本設計", "モデル比較、評価方法、初期の3用途候補"],
                ["2026-08<br/>物性絞り込みレポート", "評価軸の見直し", "2・3・4物性比較、役割分離案、p90誤差"],
                ["2026-10<br/>最新会議レポート", "実務運用への拡張", "役割分離実装、処方制約、3方針・9候補"],
                ["本資料", "横断要約", "順解析と逆解析の手法・分かること・限界"],
            ],
            [36 * mm, 54 * mm, 84 * mm],
        ),
        Spacer(1, 7 * mm),
        p("処理の全体像", STYLES["h2"]),
    ]
    flow = Table(
        [[p("配合データ", STYLES["flow"]), p("→", STYLES["flow_arrow"]), p("順解析<br/>物性予測", STYLES["flow"]), p("→", STYLES["flow_arrow"]), p("逆解析<br/>候補探索", STYLES["flow"]), p("→", STYLES["flow_arrow"]), p("試作・実測", STYLES["flow"])]],
        colWidths=[34 * mm, 9 * mm, 34 * mm, 9 * mm, 34 * mm, 9 * mm, 45 * mm],
    )
    flow.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, 0), PALE_BLUE),
                ("BACKGROUND", (2, 0), (2, 0), PALE_GREEN),
                ("BACKGROUND", (4, 0), (4, 0), PALE_YELLOW),
                ("BACKGROUND", (6, 0), (6, 0), LIGHT_GREY),
                ("BOX", (0, 0), (0, 0), 0.6, BLUE),
                ("BOX", (2, 0), (2, 0), 0.6, BLUE),
                ("BOX", (4, 0), (4, 0), 0.6, BLUE),
                ("BOX", (6, 0), (6, 0), 0.6, BLUE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.extend(
        [
            flow,
            Spacer(1, 6 * mm),
            p("逆解析は物性から配合を直接1つに決めない。同じ物性を実現する配合は複数あり、原料併用、加工性、法規、在庫、コストは物性値だけでは決められないため、複数候補を人が確認する方式にしている。", STYLES["body"]),
        ]
    )

    story.append(PageBreak())
    story.append(p("1. 順解析 - 配合から物性を予測", STYLES["h1"]))
    story.append(
        table(
            [
                ["入力", "手法", "出力"],
                ["PVC、可塑剤、充填剤、安定剤、滑剤、添加剤などの配合量", "Random Forest<br/>XGBoost<br/>LightGBM", "硬度、100%M、引張強さ、引張伸び、脆化温度、比重"],
            ],
            [58 * mm, 48 * mm, 68 * mm],
        )
    )
    story.append(Spacer(1, 6 * mm))
    story.append(p("評価に使った手法", STYLES["h2"]))
    for text in (
        "通常ランダム分割で、既存配合に近い範囲の補間性能を確認。",
        "5-fold OOFで、各行を学習に含めない予測誤差を確認。",
        "主PVC・主可塑剤のグループ分割で、未知材料系列への汎化性能を確認。",
        "特徴量重要度、SHAP、ルール候補でモデルの判断材料を確認。",
    ):
        story.append(bullet(text))
    story.append(Spacer(1, 4 * mm))
    story.append(p("LightGBMの5-fold OOF結果", STYLES["h2"]))
    story.append(
        table(
            [
                ["物性", "R²", "MAE", "p90絶対誤差"],
                ["硬度", "0.834", "3.35", "7.80"],
                ["100%モジュラス", "0.842", "1.06", "2.53"],
                ["引張強さ", "0.805", "1.57", "3.54"],
                ["引張伸び", "0.709", "23.89", "50.80"],
                ["脆化温度", "0.816", "3.11", "6.72"],
                ["比重", "0.855", "0.0217", "0.0471"],
            ],
            [64 * mm, 30 * mm, 38 * mm, 42 * mm],
        )
    )
    story.append(Spacer(1, 6 * mm))
    story.append(
        card(
            "順解析で分かったこと",
            "既存系列内では候補の方向付けに使える。一方、引張伸びは相対的に誤差が大きく、未知の材料系列では性能が低下する。予測値は実測値そのものではなく、候補間の大きな差と試作順を判断するために使う。",
            PALE_BLUE,
        )
    )

    story.append(PageBreak())
    story.append(p("2. 逆解析 - 目標に近い候補を探索", STYLES["h1"]))
    inverse_steps = [
        ["段階", "実施内容", "採用理由"],
        ["1. 起点選定", "目標用途に近い既存配合を選ぶ", "実測領域に近い場所から探索する"],
        ["2. 候補生成", "同じ主PVC・主可塑剤系列を25/50/75%で補間", "現実の材料系列から大きく外れにくい"],
        ["3. 物性予測", "順解析モデルで候補物性を予測", "同じ物差しで多数候補を比較できる"],
        ["4. 役割別評価", "最適化・管理目標・制約・監視に分離", "有望候補を一律条件で落としすぎない"],
        ["5. 候補選定", "score、Pareto、多様性、処方制約で選ぶ", "性能と実務条件を両立させる"],
        ["6. 保守確認", "p90予測幅と3方針で提示", "予測の過信を防ぎ比較軸を明確にする"],
    ]
    story.append(table(inverse_steps, [27 * mm, 77 * mm, 70 * mm]))
    story.append(Spacer(1, 6 * mm))
    story.append(p("最新の柔軟・低温用途の結果", STYLES["h2"]))
    latest = [
        ["仮想候補", "処方制約通過", "物性＋処方通過", "方針別選定", "レビュー対象"],
        ["345件", "63件", "16件", "3方針 × 5件", "9候補"],
    ]
    latest_table = table(latest, [34.8 * mm] * 5)
    latest_table.setStyle(TableStyle([("BACKGROUND", (0, 1), (-1, 1), PALE_GREEN), ("ALIGN", (0, 1), (-1, 1), "CENTER")]))
    story.append(latest_table)
    story.append(Spacer(1, 6 * mm))
    for text in (
        "性能重視: 脆化温度の低さを優先。",
        "単純処方重視: 材料数の少なさを優先。",
        "監視値重視: 比重、100%M、引張強さの参考条件からの外れを優先して小さくする。",
    ):
        story.append(bullet(text))
    story.append(Spacer(1, 4 * mm))
    story.append(
        card(
            "逆解析で分かったこと",
            "目標に近い候補、性能と処方複雑さのトレードオフ、条件変更による候補の入れ替わりを比較できる。全6物性のp90保守判定を同時通過した候補は0件であり、候補推薦は合格保証ではなく、複数候補を試作するための優先順位である。",
            PALE_YELLOW,
        )
    )

    story.append(PageBreak())
    story.append(p("3. 何を知りたいときに、どちらを使うか", STYLES["h1"]))
    story.append(
        table(
            [
                ["知りたいこと", "使う解析", "得られる答え"],
                ["この配合なら物性はどうなりそうか", "順解析", "各物性の予測値と誤差幅"],
                ["モデルはどの物性を予測しやすいか", "順解析", "R²、MAE、p90誤差、未知系列性能"],
                ["この目標に近い配合候補は何か", "逆解析", "複数候補、順位、材料系列、親配合との差"],
                ["性能と材料数のバランスはどうか", "逆解析", "性能重視・単純処方重視の候補比較"],
                ["条件変更で候補はどう変わるか", "逆解析", "候補数、順位、材料系列の変化"],
                ["製品として合格するか", "試作・実測", "加工性を含む実測結果。解析だけでは確定不可"],
            ],
            [68 * mm, 32 * mm, 74 * mm],
        )
    )
    story.append(Spacer(1, 7 * mm))
    story.append(p("関連資料と結果", STYLES["h2"]))
    story.append(
        table(
            [
                ["ファイル", "見る内容"],
                ["docs/forward_inverse_analysis_report.pdf", "順解析・逆解析の詳細な基本報告（15ページ）"],
                ["output/pdf/property_scope_meeting_report_2026-08-21.pdf", "物性数と役割を変えた比較結果（7ページ）"],
                ["output/pdf/meeting_report_2026-10-20_final.pdf", "最新の逆解析進捗と3方針候補（5ページ）"],
                ["outputs/role_based_design/candidate_review_sheet.csv", "最新9候補の採否・除外理由を記録する表"],
            ],
            [106 * mm, 68 * mm],
        )
    )
    story.append(Spacer(1, 7 * mm))
    story.append(
        card(
            "結論",
            "順解析で配合と物性の関係を予測・評価し、その誤差を把握する。逆解析で順解析を採点器として使い、目標に近い候補を複数探す。最後は試作・実測と企業側の処方判断で確定する。",
            PALE_GREEN,
        )
    )
    doc.build(story)


register_fonts()
STYLES = {
    "title": ParagraphStyle("title", fontName="JP-Bold", fontSize=25, leading=33, textColor=NAVY, alignment=TA_LEFT, spaceAfter=4 * mm),
    "subtitle": ParagraphStyle("subtitle", fontName="JP", fontSize=10.5, leading=16, textColor=GREY),
    "eyebrow": ParagraphStyle("eyebrow", fontName="JP-Bold", fontSize=9, leading=12, textColor=BLUE, spaceAfter=3 * mm),
    "h1": ParagraphStyle("h1", fontName="JP-Bold", fontSize=18, leading=25, textColor=NAVY, spaceAfter=6 * mm),
    "h2": ParagraphStyle("h2", fontName="JP-Bold", fontSize=13, leading=19, textColor=NAVY, spaceBefore=2 * mm, spaceAfter=3 * mm),
    "body": ParagraphStyle("body", fontName="JP", fontSize=9.2, leading=15, textColor=colors.HexColor("#25333D")),
    "bullet": ParagraphStyle("bullet", fontName="JP", fontSize=9.2, leading=14, leftIndent=4 * mm, firstLineIndent=-3 * mm, spaceAfter=1.5 * mm, textColor=colors.HexColor("#25333D")),
    "card_title": ParagraphStyle("card_title", fontName="JP-Bold", fontSize=11, leading=15, textColor=NAVY),
    "table_header": ParagraphStyle("table_header", fontName="JP-Bold", fontSize=7.4, leading=10, textColor=colors.white, alignment=TA_CENTER),
    "table_cell": ParagraphStyle("table_cell", fontName="JP", fontSize=7.3, leading=10, textColor=colors.HexColor("#25333D")),
    "flow": ParagraphStyle("flow", fontName="JP-Bold", fontSize=9, leading=13, textColor=NAVY, alignment=TA_CENTER),
    "flow_arrow": ParagraphStyle("flow_arrow", fontName="JP-Bold", fontSize=15, leading=18, textColor=BLUE, alignment=TA_CENTER),
}


if __name__ == "__main__":
    build()
