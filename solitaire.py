"""
Klondike Solitaire Solver

Models a standard Klondike solitaire game and uses depth-first search
with pruning to find a winning sequence of moves.
"""

import random
import copy
import sys

# ---------------------------------------------------------------------------
# Card
# ---------------------------------------------------------------------------

SUITS = ("Hearts", "Diamonds", "Clubs", "Spades")
SUIT_SYMBOLS = {"Hearts": "\u2665", "Diamonds": "\u2666", "Clubs": "\u2663", "Spades": "\u2660"}
RANKS = ("A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K")
RED_SUITS = {"Hearts", "Diamonds"}
BLACK_SUITS = {"Clubs", "Spades"}


class Card:
    """A single playing card."""

    def __init__(self, rank: str, suit: str, face_up: bool = False):
        self.rank = rank
        self.suit = suit
        self.face_up = face_up

    @property
    def value(self) -> int:
        return RANKS.index(self.rank) + 1

    @property
    def color(self) -> str:
        return "red" if self.suit in RED_SUITS else "black"

    def flip_up(self):
        self.face_up = True

    def __repr__(self):
        if not self.face_up:
            return "[##]"
        sym = SUIT_SYMBOLS[self.suit]
        r = self.rank.rjust(2)
        return f"[{r}{sym}]"

    def __eq__(self, other):
        if not isinstance(other, Card):
            return False
        return self.rank == other.rank and self.suit == other.suit

    def __hash__(self):
        return hash((self.rank, self.suit))

    def short(self) -> str:
        """Short identifier for state hashing."""
        return f"{self.rank}{self.suit[0]}"


# ---------------------------------------------------------------------------
# Deck
# ---------------------------------------------------------------------------

class Deck:
    """Standard 52-card deck."""

    def __init__(self):
        self.cards: list[Card] = []
        for suit in SUITS:
            for rank in RANKS:
                self.cards.append(Card(rank, suit))

    def shuffle(self, seed=None):
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.cards)

    def deal(self) -> Card:
        return self.cards.pop()


# ---------------------------------------------------------------------------
# Piles
# ---------------------------------------------------------------------------

class StockPile:
    """The stock (draw pile) and waste pile.  Draws 1 card at a time."""

    def __init__(self):
        self.stock: list[Card] = []
        self.waste: list[Card] = []

    def draw(self) -> bool:
        """Draw one card from stock to waste.  Returns False if stock empty."""
        if not self.stock:
            return False
        card = self.stock.pop()
        card.flip_up()
        self.waste.append(card)
        return True

    def recycle(self) -> bool:
        """Turn waste back into stock. Returns False if waste is empty."""
        if not self.waste:
            return False
        while self.waste:
            card = self.waste.pop()
            card.face_up = False
            self.stock.append(card)
        return True

    def peek_waste(self):
        return self.waste[-1] if self.waste else None

    def take_waste(self) -> Card | None:
        return self.waste.pop() if self.waste else None

    def state_key(self) -> tuple:
        stock_ids = tuple(c.short() for c in self.stock)
        waste_ids = tuple(c.short() for c in self.waste)
        return ("S", stock_ids, waste_ids)


class FoundationPile:
    """One of the four foundation piles (one per suit, A -> K)."""

    def __init__(self, suit: str):
        self.suit = suit
        self.cards: list[Card] = []

    def can_push(self, card: Card) -> bool:
        if card.suit != self.suit:
            return False
        if not self.cards:
            return card.rank == "A"
        return card.value == self.cards[-1].value + 1

    def push(self, card: Card):
        self.cards.append(card)

    def pop(self) -> Card | None:
        return self.cards.pop() if self.cards else None

    def top(self) -> Card | None:
        return self.cards[-1] if self.cards else None

    def is_complete(self) -> bool:
        return len(self.cards) == 13

    def state_key(self) -> tuple:
        return ("F", self.suit, len(self.cards))


class Column:
    """One of the seven tableau columns."""

    def __init__(self):
        self.cards: list[Card] = []

    def top_face_up_run(self) -> list[Card]:
        """Return the face-up run at the bottom of the column."""
        run = []
        for card in reversed(self.cards):
            if card.face_up:
                run.append(card)
            else:
                break
        run.reverse()
        return run

    def can_place(self, card: Card) -> bool:
        if not self.cards:
            return card.rank == "K"
        top = self.cards[-1]
        if not top.face_up:
            return False
        return top.color != card.color and top.value == card.value + 1

    def reveal_top(self):
        if self.cards and not self.cards[-1].face_up:
            self.cards[-1].flip_up()

    def state_key(self) -> tuple:
        return ("C", tuple(
            (c.short(), c.face_up) for c in self.cards
        ))


# ---------------------------------------------------------------------------
# Board / Game State
# ---------------------------------------------------------------------------

class Board:
    """Represents a full Klondike solitaire game state."""

    def __init__(self, seed=None):
        self.stock_pile = StockPile()
        self.foundations: dict[str, FoundationPile] = {
            suit: FoundationPile(suit) for suit in SUITS
        }
        self.columns: list[Column] = [Column() for _ in range(7)]
        self.move_history: list[str] = []
        self._deal(seed)

    def _deal(self, seed=None):
        deck = Deck()
        deck.shuffle(seed)
        # Deal tableau
        for col_idx in range(7):
            for row in range(col_idx + 1):
                card = deck.deal()
                if row == col_idx:
                    card.flip_up()
                self.columns[col_idx].cards.append(card)
        # Remaining cards go to stock
        for card in deck.cards:
            self.stock_pile.stock.append(card)

    def reset(self, seed=None):
        self.stock_pile = StockPile()
        self.foundations = {suit: FoundationPile(suit) for suit in SUITS}
        self.columns = [Column() for _ in range(7)]
        self.move_history = []
        self._deal(seed)

    def is_solved(self) -> bool:
        return all(f.is_complete() for f in self.foundations.values())

    def state_key(self) -> tuple:
        """Hashable snapshot of the entire game state for cycle detection."""
        return (
            self.stock_pile.state_key(),
            tuple(f.state_key() for f in self.foundations.values()),
            tuple(c.state_key() for c in self.columns),
        )

    # -- Move generators ----------------------------------------------------

    def _generate_moves(self):
        """Yield all legal moves from the current state.

        Each move is a callable that applies the move and returns a description
        string, paired with a priority (lower is better).
        """
        moves = []

        # 1. Tableau / waste -> foundation  (highest priority)
        for ci, col in enumerate(self.columns):
            if col.cards and col.cards[-1].face_up:
                card = col.cards[-1]
                fp = self.foundations[card.suit]
                if fp.can_push(card):
                    moves.append((0, self._make_col_to_foundation(ci)))

        waste_card = self.stock_pile.peek_waste()
        if waste_card:
            fp = self.foundations[waste_card.suit]
            if fp.can_push(waste_card):
                moves.append((0, self._make_waste_to_foundation()))

        # 2. Tableau -> tableau
        for src_i, src_col in enumerate(self.columns):
            run = src_col.top_face_up_run()
            if not run:
                continue
            move_card = run[0]  # top of the face-up run
            for dst_i, dst_col in enumerate(self.columns):
                if src_i == dst_i:
                    continue
                if dst_col.can_place(move_card):
                    # Skip moving a king to an empty column if it's already
                    # at the base of its column (no hidden cards beneath).
                    if move_card.rank == "K" and not dst_col.cards:
                        hidden_below = any(
                            not c.face_up for c in src_col.cards
                            if c != move_card
                        )
                        if not hidden_below:
                            continue
                    n = len(run)
                    moves.append((1, self._make_col_to_col(src_i, dst_i, n)))

        # 3. Waste -> tableau
        if waste_card:
            for ci, col in enumerate(self.columns):
                if col.can_place(waste_card):
                    moves.append((2, self._make_waste_to_col(ci)))

        # 4. Draw from stock
        if self.stock_pile.stock:
            moves.append((3, self._make_draw()))

        # 5. Recycle waste -> stock
        if not self.stock_pile.stock and self.stock_pile.waste:
            moves.append((4, self._make_recycle()))

        # Sort by priority
        moves.sort(key=lambda x: x[0])
        return [m for _, m in moves]

    # -- Move factories -----------------------------------------------------

    def _make_col_to_foundation(self, col_idx):
        def execute():
            card = self.columns[col_idx].cards.pop()
            self.foundations[card.suit].push(card)
            self.columns[col_idx].reveal_top()
            desc = f"Column {col_idx+1} -> Foundation: {card}"
            self.move_history.append(desc)
            return desc
        return execute

    def _make_waste_to_foundation(self):
        def execute():
            card = self.stock_pile.take_waste()
            self.foundations[card.suit].push(card)
            desc = f"Waste -> Foundation: {card}"
            self.move_history.append(desc)
            return desc
        return execute

    def _make_col_to_col(self, src, dst, count):
        def execute():
            cards = self.columns[src].cards[-count:]
            del self.columns[src].cards[-count:]
            self.columns[dst].cards.extend(cards)
            self.columns[src].reveal_top()
            label = " ".join(str(c) for c in cards)
            desc = f"Column {src+1} -> Column {dst+1}: {label}"
            self.move_history.append(desc)
            return desc
        return execute

    def _make_waste_to_col(self, col_idx):
        def execute():
            card = self.stock_pile.take_waste()
            self.columns[col_idx].cards.append(card)
            desc = f"Waste -> Column {col_idx+1}: {card}"
            self.move_history.append(desc)
            return desc
        return execute

    def _make_draw(self):
        def execute():
            self.stock_pile.draw()
            card = self.stock_pile.peek_waste()
            desc = f"Draw: {card}"
            self.move_history.append(desc)
            return desc
        return execute

    def _make_recycle(self):
        def execute():
            self.stock_pile.recycle()
            desc = "Recycle waste -> stock"
            self.move_history.append(desc)
            return desc
        return execute

    # -- Solver -------------------------------------------------------------

    def solve(self, max_states: int = 200_000, verbose: bool = False):
        """Attempt to solve via DFS with visited-state pruning.

        Returns the list of moves if solved, or None.
        """
        visited: set[tuple] = set()
        # Stack entries: (board_snapshot, moves_so_far)
        initial_snapshot = copy.deepcopy(self)
        initial_snapshot.move_history = []
        stack = [initial_snapshot]
        states_explored = 0

        while stack:
            board = stack.pop()
            state = board.state_key()

            if state in visited:
                continue
            visited.add(state)
            states_explored += 1

            if verbose and states_explored % 10_000 == 0:
                foundation_count = sum(
                    len(f.cards) for f in board.foundations.values()
                )
                print(
                    f"  States explored: {states_explored}, "
                    f"Foundation cards: {foundation_count}, "
                    f"Stack size: {len(stack)}"
                )

            if board.is_solved():
                if verbose:
                    print(f"  Solved after exploring {states_explored} states!")
                self.move_history = board.move_history
                return board.move_history

            if states_explored >= max_states:
                if verbose:
                    print(
                        f"  Reached state limit ({max_states}). "
                        "Giving up on this deal."
                    )
                return None

            # Generate moves on this board, then for each move, deepcopy
            # the board and regenerate+execute the corresponding move on
            # the copy.  Reversed so highest-priority moves are explored
            # first (they end up on top of the DFS stack).
            moves = board._generate_moves()
            for i in reversed(range(len(moves))):
                child = copy.deepcopy(board)
                child_moves = child._generate_moves()
                if i < len(child_moves):
                    child_moves[i]()
                    child_state = child.state_key()
                    if child_state not in visited:
                        stack.append(child)

        if verbose:
            print(f"  Exhausted search space ({states_explored} states). No solution.")
        return None

    # -- Display ------------------------------------------------------------

    def display(self):
        """Print the current board state to stdout."""
        # Stock + Waste
        stock_count = len(self.stock_pile.stock)
        waste_top = self.stock_pile.peek_waste()
        waste_str = str(waste_top) if waste_top else "[  ]"
        print(f"Stock ({stock_count:>2}): {waste_str}    ", end="")

        # Foundations
        print("Foundations: ", end="")
        for suit in SUITS:
            fp = self.foundations[suit]
            top = fp.top()
            sym = SUIT_SYMBOLS[suit]
            if top:
                print(f" {top}", end="")
            else:
                print(f" [ {sym}]", end="")
        print()
        print("-" * 60)

        # Tableau
        max_height = max((len(c.cards) for c in self.columns), default=0)
        # Header
        print("  ".join(f" Col{i+1}" for i in range(7)))
        for row in range(max_height):
            parts = []
            for col in self.columns:
                if row < len(col.cards):
                    parts.append(str(col.cards[row]))
                else:
                    parts.append("     ")
            print("  ".join(parts))
        if max_height == 0:
            print("  (empty tableau)")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    seed = None
    max_states = 200_000
    verbose = True

    # Parse simple CLI args
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] in ("--seed", "-s") and i + 1 < len(args):
            seed = int(args[i + 1])
            i += 2
        elif args[i] in ("--max-states", "-m") and i + 1 < len(args):
            max_states = int(args[i + 1])
            i += 2
        elif args[i] in ("--quiet", "-q"):
            verbose = False
            i += 1
        elif args[i] in ("--help", "-h"):
            print("Usage: solitaire.py [--seed N] [--max-states N] [--quiet]")
            print()
            print("Options:")
            print("  --seed N, -s N         Random seed for the deal")
            print("  --max-states N, -m N   Max states to explore (default 200000)")
            print("  --quiet, -q            Suppress progress output")
            return
        else:
            print(f"Unknown argument: {args[i]}")
            return

    if seed is None:
        seed = random.randint(0, 999_999)

    print(f"=== Klondike Solitaire Solver ===")
    print(f"Seed: {seed}")
    print()

    board = Board(seed=seed)
    board.display()

    print("Solving...")
    solution = board.solve(max_states=max_states, verbose=verbose)

    if solution:
        print(f"\nSolution found! ({len(solution)} moves)")
        print("-" * 40)
        for i, move in enumerate(solution, 1):
            print(f"  {i:>3}. {move}")
        print()
        print("All 52 cards moved to foundations. Game won!")
    else:
        print("\nNo solution found within the search limit.")
        print("Try a different seed or increase --max-states.")


if __name__ == "__main__":
    main()
