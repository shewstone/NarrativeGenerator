"""Human-authored seed passages with gold event spans and normalization."""

from dataclasses import dataclass

from narrative_engine.evaluation.extraction_quality import GoldEpisode


@dataclass(frozen=True)
class ExtractionCase:
    name: str
    text: str
    episodes: tuple[GoldEpisode, ...]


def _case(name: str, parts: list[tuple[str, str, int]]) -> ExtractionCase:
    text = "\n\n".join(part[0] for part in parts)
    episodes = []
    cursor = 0
    for episode_text, scope_id, year in parts:
        start = text.index(episode_text, cursor)
        end = start + len(episode_text)
        episodes.append(GoldEpisode(start, end, scope_id, year))
        cursor = end
    return ExtractionCase(name=name, text=text, episodes=tuple(episodes))


EXTRACTION_CASES = (
    _case(
        "french_revolution_turns",
        [
            (
                "In May 1789 Louis XVI summoned the Estates-General. "
                "Disputes over voting led the Third Estate to declare itself the National Assembly.",
                "french_revolution",
                1789,
            ),
            (
                "On 14 July 1789 a Parisian crowd captured the Bastille, destroying a royal "
                "fortress and accelerating the transfer of political initiative away from the crown.",
                "french_revolution",
                1789,
            ),
            (
                "In September 1792 the Convention abolished the monarchy and proclaimed a republic. "
                "The former king was executed the following January.",
                "french_revolution",
                1792,
            ),
        ],
    ),
    _case(
        "china_party_trajectory",
        [
            (
                "The Chinese Communist Party was founded in Shanghai in 1921 by a small network of Marxist organizers.",
                "chinese_communist_party",
                1921,
            ),
            (
                "After the alliance with the Nationalists collapsed in 1927, Communist organizations "
                "were driven from the cities and rebuilt around rural bases.",
                "chinese_communist_party",
                1927,
            ),
            (
                "In 1949 Communist victory in the civil war allowed Mao Zedong to proclaim the "
                "People's Republic of China.",
                "chinese_communist_party",
                1949,
            ),
        ],
    ),
    _case(
        "movement_and_reaction",
        [
            (
                "British suffrage organizations expanded petitioning and public meetings during "
                "the late nineteenth century.",
                "uk_womens_suffrage_movement",
                1890,
            ),
            (
                "The Women's Social and Political Union adopted militant tactics after 1903, "
                "prompting arrests, force-feeding, and an organized anti-suffrage response.",
                "uk_womens_suffrage_movement",
                1903,
            ),
            (
                "The Representation of the People Act 1918 enfranchised many British women, while "
                "equal voting terms were not achieved until 1928.",
                "uk_womens_suffrage_movement",
                1918,
            ),
        ],
    ),
)
