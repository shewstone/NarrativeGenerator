"""Controlled actor-role vocabulary (T2, docs/tickets/T2-structural-render-hardening.md).

Roles name STRUCTURAL POSITIONS, not costumes (design doc Sec 10.1): a
"court" is any power center where proximity to a principal outweighs formal
office — an administration's inner circle, a founder-CEO's kitchen cabinet,
a politburo. Admission test per role: would two episodes a thousand years
apart share this token in a way that reveals shape?

The vocabulary is deliberately small (~50, never past ~80): roles that
appear once never form clusters, and the entire point is forcing analogous
actors across millennia into the same token. It is versioned like the
taxonomy; actors that fit no role well accumulate as residue
(taxonomy.residue), which is the input signal for the Sec 10.5
vocabulary-evolution loop.

Roles can be filled by collective actors (an investor syndicate as
KINGMAKER, a movement as PRETENDER) — extraction prompts say so explicitly,
or the vocabulary only spots roles wearing individual faces (Sec 10.1a).
"""

from enum import Enum

CURRENT_ROLE_VOCAB_VERSION = "role-v0.2.0"


class ActorRole(str, Enum):
    """Controlled vocabulary of structural actor positions."""

    # --- Domain-neutral group dynamics ---
    INCUMBENT = "incumbent"                          # established holder defending position
    CHALLENGER = "challenger"                        # actor seeking to displace an incumbent
    COALITION = "coalition"                          # aligned actors pooling support/capability
    CONSTITUENCY = "constituency"                    # base whose support grants capacity/legitimacy
    DISSENTER = "dissenter"                          # insider openly departing from group consensus
    SUCCESSOR = "successor"                          # recipient of role, authority, or stewardship
    MEDIATOR = "mediator"                            # bridges opposed actors toward coordination
    GATEKEEPER = "gatekeeper"                        # controls access to membership/resources/audience
    BROKER = "broker"                                # connects otherwise separated actors or groups
    INNOVATOR = "innovator"                          # introduces a practice that changes possibilities
    CUSTODIAN = "custodian"                          # preserves memory, norms, or capability through change

    # --- Political-economy (Sec 3.3 seed set) ---
    RISING_POWER = "rising_power"                    # challenger gaining relative capability
    INCUMBENT_HEGEMON = "incumbent_hegemon"          # order-setting power defending position
    DECLINING_POWER = "declining_power"              # losing relative position, still armed
    CENTRAL_AUTHORITY = "central_authority"          # the state/crown/executive as actor
    REGULATOR = "regulator"                          # rule-setter over markets/conduct
    CENTRAL_BANK = "central_bank"                    # lender of last resort / monetary authority
    CREDITOR_CLASS = "creditor_class"                # holders of claims, pro-hard-money
    DEBTOR_CLASS = "debtor_class"                    # obligors, pro-inflation/relief
    FINANCIER = "financier"                          # intermediary funding both sides
    SPECULATOR = "speculator"                        # leveraged bet-taker on continuation
    MERCHANT_CLASS = "merchant_class"                # trading interest, pro-openness
    LANDED_ELITE = "landed_elite"                    # holders of fixed productive assets
    INDUSTRIAL_ELITE = "industrial_elite"            # holders of productive capital
    LABOR_MOVEMENT = "labor_movement"                # organized wage-earners as collective actor
    COUNTER_ELITE = "counter_elite"                  # elite-credentialed outsiders vs incumbents
    ASPIRANT_ELITE = "aspirant_elite"                # credentialed aspirants exceeding available slots
    POPULIST_TRIBUNE = "populist_tribune"            # leader channeling mass grievance vs elites
    TECHNOCRAT = "technocrat"                        # expertise-legitimated administrator

    # --- Political-dramatic (court/succession family, Sec 10.1) ---
    SOVEREIGN = "sovereign"                          # the principal whose favor is the currency
    HEIR_APPARENT = "heir_apparent"                  # designated successor, target of factions
    USURPER = "usurper"                              # seizes the principal position by force/coup
    PRETENDER = "pretender"                          # rival claimant to the principal position
    REGENT = "regent"                                # rules in the sovereign's name
    KINGMAKER = "kingmaker"                          # can install but not occupy the throne
    COURT_FAVORITE = "court_favorite"                # power via proximity, not office; falls fast
    COURTIER_FACTION = "courtier_faction"            # coalition competing for proximity
    PURGED_FACTION = "purged_faction"                # losers of an internal power struggle
    PALACE_GUARD = "palace_guard"                    # coercive force with kingmaking leverage
    PROVINCIAL_MAGNATE = "provincial_magnate"        # regional power semi-independent of center

    # --- Proppian-functional (Sec 10.4) ---
    PATRON = "patron"                                # provides resources/protection to protagonist
    DISPATCHER = "dispatcher"                        # sends the protagonist on the undertaking
    HELPER = "helper"                                # aids at critical junctures
    FALSE_HERO = "false_hero"                        # claims credit/legitimacy until unmasked
    TRAITOR_WITHIN = "traitor_within"                # trusted insider who defects
    RIVAL_CLAIMANT = "rival_claimant"                # competes for the same prize

    # --- Social-mobility ---
    PARVENU = "parvenu"                              # newly risen, unaccepted by old elite
    DECLINING_HOUSE = "declining_house"              # old elite losing material base
    DISPOSSESSED_CLASS = "dispossessed_class"        # losers of an economic transition
    RISING_CLASS = "rising_class"                    # gaining material base, denied status

    # --- Religious / ideological ---
    PROPHET_FIGURE = "prophet_figure"                # moral authority from outside institutions
    HERESIARCH = "heresiarch"                        # founder of a rival doctrine within a tradition
    ORTHODOX_HIERARCHY = "orthodox_hierarchy"        # incumbent doctrinal authority
    TRUE_BELIEVER_MOVEMENT = "true_believer_movement"  # mass movement organized around doctrine
    APOSTATE = "apostate"                            # high-profile defector from the doctrine
    REFORMER = "reformer"                            # changes the institution to save it
    REACTIONARY = "reactionary"                      # restores the prior order

    # --- Conflict / external ---
    INVADER = "invader"                              # external force entering the scope
    DEFENDER = "defender"                            # organizes resistance to the invader
    INSURGENT = "insurgent"                          # internal armed challenger
    MERCENARY = "mercenary"                          # loyalty for hire, switches sides
    NEUTRAL_ARBITER = "neutral_arbiter"              # third party whose alignment decides outcomes
    CLIENT_STATE = "client_state"                    # dependent polity of a greater power

    # --- Fallback ---
    BYSTANDER_MASS = "bystander_mass"                # the population as acted-upon aggregate


# Render token for actors whose free-text role failed the fit floor: free
# text must NEVER reach the structural embedding, so unresolved actors all
# share one token (and count as residue, taxonomy.residue).
UNRESOLVED_ACTOR_TOKEN = "unresolved_actor"

ROLE_VALUES = {role.value for role in ActorRole}


def is_known_role(value: str) -> bool:
    return value in ROLE_VALUES
