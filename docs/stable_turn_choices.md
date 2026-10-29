# Source end-turn choices

Well-Laid Plans installs a persistent owner power. Its optional retention choice
now runs at `BeforeFlushLate`, after ordinary orb/hand effects and `BeforeFlush`
cost changes. Burn and Ethereal cards have already left the hand. Cards already
retaining are excluded, and the prompt is omitted when the source says not to
flush, including Runic Pyramid, first-turn Ringing Triangle and Retain Hand.
Only selected physical cards receive single-turn Retain.

Stampede is a persistent power with a sequential AutoPost loop. Each iteration
uses the source Shuffle stream to sample the current eligible hand, then waits
for the autoplay, child choices and replays before selecting again. This matters
when Dagger Throw draws another Attack and asks which card to discard. Power
listeners also precede physical-card AutoPost listeners such as Howl from Beyond.

The engine persists whether an ongoing end turn is suspended in AutoPost or at
the completed BeforeFlush boundary. Choosing the last card, or skipping an
optional choice, automatically resumes the same end turn. It no longer requires
another End Turn action, and resuming does not repeat earlier orb/hand effects.

Source paths, hashes and finite method facts are in
`tests/fixtures/source_turn_choices.json`. Public fixtures cover reordered hand
effects, optional skips, save/reload, sequential autoplay choices, stream draw
counts, lethal autoplay and the Pael's Eye interaction. These checks are not an
installed-game comparison or an exhaustive ordering certificate.
